
#%%

import numpy as np
import pandas as pd
import pickle
import glob
import rasterio as rd
import statsmodels.formula.api as smf
import dataretrieval.nwis as nwis

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors

save_directory = 'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/'

#%% Saturation on hillshade

path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/'
name = 'LSDTT/baltimore2015_DR1_hs.bil' # Druids Run hillshade

# path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/'
# name = 'LSDTT/baltimore2015_BR_hs.bil' # Baisman Run hillshade

paths = glob.glob(path + "saturation/transects_*.csv")

for i in range(len(paths)):
    src = rd.open(path + name) # hillshade
    # df = pd.read_csv(path + name_pts) # sampled pts
    df = pd.read_csv(paths[i]) # sampled pts
    A = [True if X in ['N', 'Ys', 'Yp', 'Yf'] else False for X in df['Name']]
    df = df[A]

    bounds = src.bounds
    Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
    proj = src.crs
    utm = 18

    fig = plt.figure(figsize=(5,6)) #(6,5)
    ax = fig.add_subplot(1, 1, 1) #, projection=ccrs.UTM(utm)

    cols = {'N':"peru", 'Ys':"dodgerblue", 'Yp':"blue", 'Yf':"navy"}

    grouped = df.groupby('Name')
    for key, group in grouped:
        group.plot(ax=ax, 
                    kind='scatter', 
                    x='X', 
                    y='Y', 
                    label=key, 
                    color=cols[key], 
                    )
    cs = ax.imshow(src.read(1), 
                    cmap='binary', 
                    extent=Extent, 
                    vmin=100,
                    origin="upper")
    ax.set_xlim((341200, 341500)) # DR bounds
    ax.set_ylim((4.36490e6,4.36511e6)) # DR Bounds
    # ax.set_xlim((354550, 355100)) # PB bounds
    # ax.set_ylim((4.37135e6,4.37225e6)) # PB Bounds

    ax.set_title(df.BeginTime[1][0:10])
    plt.tight_layout()
    plt.savefig(f'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/DruidRun_sat_{df.BeginTime[1][0:10]}.png')
    # plt.savefig(f'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/BaismanRun_sat_{df.BeginTime[1][0:10]}.png')

    plt.show()


#%% TI vs saturation

path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/'
TIfile = "LSDTT/baltimore2015_DR1_TIfiltered.tif" # Druids Run
curvfile = "LSDTT/baltimore2015_DR1_CURV.bil" # Druids Run

# path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/'
# TIfile = "LSDTT/baltimore2015_BR_TIfiltered.tif" # Baisman Run
# curvfile = "LSDTT/10m_window/baltimore2015_BR_CURV.bil" # Baisman Run

paths = glob.glob(path + "saturation/transects_*.csv")

for i in range(len(paths)):

    
    df = pd.read_csv(paths[i]) # sampled pts

    # get the saturation value right
    A = [True if X in ['N', 'Ys', 'Yp', 'Yf'] else False for X in df['Name']]
    df = df[A]
    sat_val_dict = {'N':0, 'Ys':1, 'Yp':2, 'Yf':3}
    df['sat_val'] = df['Name'].apply(lambda x: sat_val_dict[x])

    coords = [(x,y) for x, y in zip(df['X'], df['Y'])]

    # open TI filtered and extract at coordinates
    tis = rd.open(path+TIfile)
    df['TI_filtered'] = [x for x in tis.sample(coords)]
    tis.close()

    # open curv extract at coordinates 
    cur = rd.open(path+curvfile)
    df['curv'] = [x for x in cur.sample(coords)]
    cur.close()

    fig, ax = plt.subplots()
    sc = ax.scatter(df['TI_filtered'], 
                    df['sat_val'] + 0.05*np.random.randn(len(df)), 
                    c=df['curv'], 
                    cmap='coolwarm', 
                    norm=colors.CenteredNorm())
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(['N', 'Ys', 'Yp', 'Yf'])
    ax.set_xlabel('TI')
    ax.set_title(df.BeginTime[1][0:10])
    fig.colorbar(sc, label='curvature')
    # plt.savefig(f'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/DruidRun_sat_TI_{df.BeginTime[1][0:10]}.png')
    # plt.savefig(f'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/BaismanRun_sat_TI_{df.BeginTime[1][0:10]}.png')
    plt.show()


#%% assemble all saturation dataframes

# site = 'DR'
# path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/'
# TIfile = "LSDTT/baltimore2015_DR1_TIfiltered.tif" # Druids Run
# curvfile = "LSDTT/baltimore2015_DR1_CURV.bil" # Druids Run

site = 'BR'
path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/'
TIfile = "LSDTT/baltimore2015_BR_TIfiltered.tif" # Baisman Run
curvfile = "LSDTT/10m_window/baltimore2015_BR_CURV.bil" # Baisman Run

dfs = []
paths = glob.glob(path + "saturation/transects_*.csv")
for pts_path in paths:

    # load and remove non-sat points
    df = pd.read_csv(pts_path) # sampled pts
    A = [True if X in ['N', 'Ys', 'Yp', 'Yf'] else False for X in df['Name']]
    df = df[A]

    # get date
    datetime = pd.to_datetime(df.BeginTime)
    df['date'] = datetime.dt.date
    df['datetime_start'] = datetime.iloc[0]
    
    # add discharge to df 

    dfs.append(df)
dfall = pd.concat(dfs, axis=0)

# %% Druids Run: Load Q

q_path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data_processed/'

# load continuous discharge (we'll use it to fill a day where a dilution gage was not done)
area_DR = 107e4 #m2
q_DR_cont = pd.read_csv(q_path+'DruidRun_discharge_15min_2022_3-2022_9.csv', 
                        parse_dates=[0],
                        infer_datetime_format=True)
q_DR_cont.set_index('Datetime', inplace=True)
q_DR_cont['Q m/d'] = q_DR_cont['Q m3/s'] * 3600 * 24 * (1/area_DR) # sec/hr * hr/d * 1/m2

# load dilution gaged Q for Druids Run
q_DR = pickle.load(open(q_path+'discharge_DR.p', 'rb'))
dfq = pd.DataFrame.from_dict(q_DR, orient='index', dtype=None, columns=['Q']) # Q in L/s
t1 = pd.Timestamp('2022-04-27 14:45:00')
dfq.loc[t1] = q_DR_cont['Q m3/s'].loc[t1] * 1000
dfq = dfq.sort_index()
dfq['datetime'] = dfq.index
dfq['Q m/d'] = dfq['Q'] * (1/1000) * 3600 * 24 * (1/area_DR) # m3/liter * sec/hr * hr/d * 1/m2
dfq['date'] = dfq['datetime'].dt.date

# load dilution gaged Q for Druids Run Upper Watershed
area_DRUW = 7.1e4 #m2
q_UG = pickle.load(open(q_path+'discharge_UG.p', 'rb'))
dfqug = pd.DataFrame.from_dict(q_UG, orient='index', dtype=None, columns=['Q']) # Q in L/s
dfqug['datetime'] = dfqug.index
dfqug['date'] = dfqug['datetime'].dt.date
dfqug['Q m/d'] = dfqug['Q'] * (1/1000) * 3600 * 24 * (1/area_DRUW) # m/liter * sec/hr * 1/m2

# %% Baisman Run: Load Q

site_BR = '01583580'
site_PB = '01583570'

dfq = nwis.get_record(sites=site_BR, service='iv', start='2022-08-01', end='2023-02-10')
dfqug = nwis.get_record(sites=site_PB, service='iv', start='2022-08-01', end='2023-02-10')

# dfq.to_csv(path+'dfq.csv')
# dfqug.to_csv(path+'dfqug.csv')

#%% Baisman run: process Q

# area normalized discharge
area_BR = 381e4 #m2
dfq['Q m/d'] = dfq['00060']*0.3048**3 * 3600 * 24 * (1/area_BR) #m3/ft3 * sec/hr * hr/d * 1/m2
dfq.drop(columns=['00060', 'site_no', '00065', '00065_cd'], inplace=True)

area_PB = 37e4 #m2
dfqug['Q m/d'] = dfqug['00060']*0.3048**3 * 3600 * 24 * (1/area_PB) #m3/ft3 * sec/hr * hr/d * 1/m2
dfqug.drop(columns=['00060', 'site_no', '00065', '00065_cd'], inplace=True)

# index from string to datetime
dfq['datetime'] = pd.to_datetime(dfq.index, utc=True)
dfq.set_index('datetime', inplace=True)

dfqug['datetime'] = pd.to_datetime(dfqug.index, utc=True)
dfqug.set_index('datetime', inplace=True)

# get the start times of every saturation survey
times = dfall.datetime_start.unique()
times = [time.round('5min') for time in times]

# isolate discharge at those times
dfq = dfq.loc[times]
dfq['datetime'] = dfq.index
dfq['date'] = dfq['datetime'].dt.date

dfqug = dfqug.loc[times]
dfqug['datetime'] = dfqug.index
dfqug['date'] = dfqug['datetime'].dt.date

#%% Add to saturation dataframe

# add filtered TI points
tis = rd.open(path+TIfile)
coords = [(x,y) for x, y in zip(dfall['X'], dfall['Y'])]
dfall['TI_filtered'] = [x[0] for x in tis.sample(coords)]
tis.close()

# add sat val
sat_val_dict = {'N':0, 'Ys':1, 'Yp':2, 'Yf':3}
dfall['sat_val'] = dfall['Name'].apply(lambda x: sat_val_dict[x])

# add discharge
dfnew = dfall.merge(dfq, on='date', how='left')
# dfnew = dfall.merge(dfqug, on='date', how='left')

dfnew['Q'] = dfnew['Q m/d'] #* 1000
dfnew.drop(columns=['OID_', 'BeginTime', 'Unnamed: 0', 'FolderPath'], inplace=True, errors='ignore')

#%% scatter TI-sat

fig, ax = plt.subplots()

cnt = -2
grouped = dfnew.groupby('date')
for key, group in grouped:
    ax.scatter(group['TI_filtered'], 
                group['sat_val'] + 0.1*cnt,
                c=group['Q'],
                vmin=min(dfnew['Q']),
                vmax=max(dfnew['Q']),
                cmap='plasma',
                )
    cnt += 1
ax.set_yticks([0,1,2,3])
ax.set_yticklabels(['N', 'Ys', 'Yp', 'Yf'])
ax.set_xlabel('TI')
plt.show()


#%% Logistic regression: sat_bin ~ logTIQ

# logTIQ = TI + log(Q) because TI is already log
dfnew['logTIQ'] = dfnew['TI_filtered'] + np.log(dfnew['Q'])
dfnew['sat_bin'] = (dfnew['sat_val'] > 0) * 1

# logit using TI and Q as predictors
model = smf.logit('sat_bin ~ logTIQ', data=dfnew).fit()

# check model performance
print(model.summary())
with open(save_directory+'summary_%s_logTIQ.txt'%site, 'w') as fh:
    fh.write(model.summary().as_text())

# predict in sample
in_sample = pd.DataFrame({'prob':model.predict()})

#%% Logistic regression: sat_bin ~ TI_filtered + logQ

# make saturation into a binary field
dfnew['sat_bin'] = (dfnew['sat_val'] > 0) * 1
dfnew['logQ'] = np.log(dfnew['Q'])

# logit using TI and Q as predictors
model = smf.logit('sat_bin ~ TI_filtered + logQ', data=dfnew).fit()

# check model performance
print(model.summary())
with open(save_directory+'summary_%s_logTI_logQ.txt'%site, 'w') as fh:
    fh.write(model.summary().as_text())

# predict in sample
in_sample = pd.DataFrame({'prob':model.predict()})

#%% get TI values

if site=='DR':

    basin_name = 'LSDTT/baltimore2015_DR1_AllBasins.bil'
    bsn = rd.open(path + basin_name)
    basin = bsn.read(1) > 0 

    TIfile = "LSDTT/baltimore2015_DR1_TIfiltered.tif" # Druids Run
    tif = rd.open(path+TIfile)
    TI = tif.read(1).astype(float)
    TI = TI[basin]

if site == 'BR':

    basin_name = 'LSDTT/baltimore2015_BR_AllBasins.bil'
    bsn = rd.open(path + basin_name)
    basin = bsn.read(1) > 0 

    TIfile = "LSDTT/baltimore2015_BR_TIfiltered.tif"
    tif = rd.open(path+TIfile)
    TI = tif.read(1).astype(float)
    TI = TI[basin]

#%% Predict out of sample, and plot with TI CDF

TI1 = np.linspace(1,12,100)
Q_all = np.geomspace(dfnew['Q'].min(),dfnew['Q'].max(),5)
# Q_all = np.linspace(0.02,0.5,5)
fig, ax = plt.subplots()

for i, Q in enumerate(Q_all):
    # pred = model.predict(exog=dict(TI_filtered=TI1, Qinv=1/Q*np.ones_like(TI1)))
    pred = model.predict(exog=dict(TI_filtered=TI1, logQ=np.log(Q)*np.ones_like(TI1)))
    # pred = model.predict(exog=dict(logTIQ=TI1 + np.log(Q)))

    ax.plot(TI1, pred, color=cm.viridis(Q/max(Q_all)), label='Q=%.2f mm/d'%(Q*1000))
ax.axvspan(dfnew['TI_filtered'].min(), dfnew['TI_filtered'].max(), alpha=0.2, color='r')
ax.set_ylim(-0.05,1.05)
ax.set_xlim((0.5,10))
ax.legend(frameon=False, loc='upper left')
ax.set_xlabel('TI')
ax.set_ylabel(r'Modeled P$(saturated)$')

ax1 = ax.twinx()
ax1.plot(np.sort(TI), np.linspace(0,1,len(TI)), color='k', linewidth=1, label='CDF')
ax1.set_xlim((0.5,10))
ax1.set_ylim(-0.05,1.05)
ax1.set_ylabel(r'P$(TI \leq TI_x)$')
ax1.legend(frameon=False, loc='lower right')
# plt.savefig(save_directory+'pred_sat_ti_%s_logTIQ.png'%site)
plt.savefig(save_directory+'pred_sat_ti_%s_logTI_logQ.png'%site)


#%% calulate a transmissivity: logTIQ method

Tmean = lambda b0, b1: np.exp((-b0)/b1)

covs = model.cov_params()
means = model.params

samples = np.random.multivariate_normal(means, covs, size=100000)

Tcalc = Tmean(samples[:,0], samples[:,1])
T_median = np.median(Tcalc)
T_lq = np.percentile(Tcalc, 25)
T_uq = np.percentile(Tcalc, 75)

dfT = pd.DataFrame({'Trans. med [m2/d]':T_median,
              'Trans. lq [m2/d]': T_lq,
              'Trans. uq [m2/d]': T_uq,
              'b0':means[0],
              'b1':means[1]},
              index=[0]
              )
dfT.to_csv(save_directory+'transmissivity_%s_logTIQ.csv'%site)


#%% calulate a transmissivity: logTI + logQ method

# note, this is not well-behaved. The estimation does not converge to a predictable value using this approach

def calc_transmissivity(a0, a1, a2, Qbar, Q=1):
    b0 = a0 - (a2-a1)*np.log(Qbar)
    b1 = a1
    bstar = a2 - a1

    coef = np.exp(-b0/b1)
    q_exponent = -bstar/b1

    T = coef * (Q/Qbar)**q_exponent

    return T, coef, q_exponent


covs = model.cov_params()
means = model.params
Qbar = np.mean(dfnew['Q'])


samples = np.random.multivariate_normal(means, covs, size=500000)

Tcalc, Ccalc, Ecalc = calc_transmissivity(samples[0], samples[1], samples[2], Qbar, Q=Qbar)

T_median = np.median(Tcalc)
T_lq = np.percentile(Tcalc, 25)
T_uq = np.percentile(Tcalc, 75)

c_median = np.median(Ccalc)
c_lq = np.percentile(Ccalc, 25)
c_uq = np.percentile(Ccalc, 75)

e_median = np.median(Ecalc)
e_lq = np.percentile(Ecalc, 25)
e_uq = np.percentile(Ecalc, 75)

dfT = pd.DataFrame({'Trans. med [m2/d]':T_median,
              'Trans. lq [m2/d]': T_lq,
              'Trans. uq [m2/d]': T_uq,
              'coef med': c_median,
              'coef lq': c_lq,
              'coef uq': c_uq,
              'expon med': e_median,
              'expon lq': e_lq,
              'expon lq': e_uq,
              'a0':means[0],
              'a1':means[1],
              'a1':means[2]},
              index=[0]
              )
dfT.to_csv(save_directory+'transmissivity_%s_logTI_logQ.csv'%site, float_format="%.3f")

#%% ##### other stuff

# cumulative TI-sat
norm = colors.Normalize(vmin=min(dfnew['Q']), vmax=max(dfnew['Q']))

fig, ax = plt.subplots()
grouped = dfnew.groupby('date')
for key, group in grouped:

    group1 = group.sort_values(by='TI_filtered')
    group1['sat_bin'] = group1['sat_val'] > 0
    group1['sat_cum'] = np.cumsum(group1['sat_bin'])/len(group1)

    ax.plot(group1['TI_filtered'],
            group1['sat_cum'],
            color=cm.plasma(norm(group1['Q'].iloc[0])))
ax.set_ylabel('Proportion Saturated')
ax.set_xlabel(r'$TI \leq TI_x$')
plt.savefig(save_directory+'cumulative_sat_DR.png')

# predict odds of saturation in sample with the model
dfnew['pred_prob'] = model.predict()

fig, ax = plt.subplots()
grouped = dfnew.groupby('date')
for key, group in grouped:
    ax.scatter(group['TI_filtered'], 
                group['pred_prob'],
                c=group['Q'],
                vmin=min(dfnew['Q']),
                vmax=max(dfnew['Q']),
                cmap='plasma',
                )


#%% make a positive ratio plot for model prediction

# thresholds = np.linspace(0.2,0.8,20)
# thresholds = np.linspace(0.05,0.6,20)
thresholds = np.linspace(0.05,0.3,20)
sens = np.zeros_like(thresholds)
spec = np.zeros_like(thresholds)
for i, thresh in enumerate(thresholds):
    in_sample['pred_label'] = (in_sample['prob']>thresh).astype(int)
    cs = pd.crosstab(in_sample['pred_label'],dfnew['sat_bin'])
    sens[i] = cs[1][1]/(cs[1][1] + cs[1][0])
    spec[i] = cs[0][0]/(cs[0][0] + cs[0][1])


fig, ax = plt.subplots()
sc = ax.scatter(1-spec, sens, c=thresholds)
ax.axline([0,0], [1,1])
ax.set_xlabel('False Positive Ratio')
ax.set_ylabel('True Positive Ratio')
fig.colorbar(sc, label='threshold')
plt.savefig(save_directory+'sens_spec_%s.png'%site)

fig, ax = plt.subplots()
sc = ax.scatter(thresholds, (1-spec)+sens, c=1-spec)
ax.set_xlabel('Threshold')
ax.set_ylabel('Total Positive Ratio')
fig.colorbar(sc, label='False Positive Ratio')

#%% various TI_crit things


pcrit = 0.5
rhocrit = lambda pcrit: np.log(pcrit/(1-pcrit))
b0 = model.params[0]
b1 = model.params[1]
# b2 = model.params[2]

# # for the TI + 1/Q model
# log odds = b0 + b1*TI + b2*(1/Q)
# TIcrit = lambda Q, pcrit: ((rhocrit(pcrit) - b0) - b2 * (1 / Q)) / b1
# Tmean = np.exp(np.mean(TIcrit(Q_all, pcrit)-np.log(1/(Q_all/1000))))


# for the TI * Q model
TIcrit = lambda Q, pcrit: (rhocrit(pcrit) - b0)/ b1 - np.log(Q)
Tmean = np.exp((-b0)/b1) #np.exp((rhocrit(pcrit)-b0)/b1)

# if you don't eliminate rhocrit, you get a critical TI that depends on pcrit and Q

pcrit_all = np.linspace(0.15, 0.65, 25)
plt.figure()
plt.scatter(pcrit_all, TIcrit(np.min(Q_all), pcrit_all))
plt.scatter(pcrit_all, TIcrit(np.mean(Q_all), pcrit_all))
plt.scatter(pcrit_all, TIcrit(np.max(Q_all), pcrit_all))

# not sure what this was
TIsort = np.sort(TI)
cdf = np.linspace(0,1,len(TI))

CDF = lambda TIc: np.array([cdf[(np.abs(TIsort - tic)).argmin()] for tic in TIc])

Q_all = np.geomspace(dfnew['Q'].min(),dfnew['Q'].max(),50)
plt.figure()
plt.scatter(Q_all/1000, TIcrit(Q_all)-np.log(1/(Q_all/1000)))

# %%
