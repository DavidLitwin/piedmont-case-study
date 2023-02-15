
#%%

import numpy as np
import pandas as pd
import pickle
import glob
import rasterio as rd
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
from scipy.ndimage import gaussian_filter
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

# path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/'
# TIfile = "LSDTT/baltimore2015_DR1_TIfiltered.tif" # Druids Run
# curvfile = "LSDTT/baltimore2015_DR1_CURV.bil" # Druids Run

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

# discharge from Druids Run (DR) and DR Upper Gage (UG)
q_path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data_processed/'
q_DR = pickle.load(open(q_path+'discharge_DR.p', 'rb'))
q_UG = pickle.load(open(q_path+'discharge_UG.p', 'rb'))

dfq = pd.DataFrame.from_dict(q_DR, orient='index', dtype=None, columns=['Q'])
dfq['datetime'] = dfq.index
dfq['date'] = dfq['datetime'].dt.date
dfq.set_index('datetime', inplace=True)

dfqug = pd.DataFrame.from_dict(q_UG, orient='index', dtype=None, columns=['Q'])
dfqug['datetime'] = dfqug.index
dfqug['date'] = dfqug['datetime'].dt.date
dfqug.set_index('datetime', inplace=True)

# %% Baisman Run: Load Q

site_BR = '01583580'
site_PB = '01583570'

dfq = nwis.get_record(sites=site_BR, service='iv', start='2022-01-01', end='2023-02-10')
dfqug = nwis.get_record(sites=site_PB, service='iv', start='2022-01-01', end='2023-02-10')


#%% Baisman run: process Q

# area normalized discharge
area_BR = 381e4 #m2
dfq['Q mm/hr'] = dfq['00060']*0.3048**3*3600*1000/area_BR
dfq.drop(columns=['00060', 'site_no', '00065', '00065_cd'], inplace=True)

area_PB = 37e4 #m2
dfqug['Q mm/hr'] = dfqug['00060']*0.3048**3*3600*1000/area_PB
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
dfnew = dfall.merge(dfqug, on='date', how='left')
# dfnew = dfall.merge(dfq, on='date', how='left')
dfnew['Q mm/hr'].fillna(0.0, inplace=True)
dfnew['Q'] = dfnew['Q mm/hr']
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


#%% logistic regression with statsmodels

# make saturation into a binary field
dfnew['sat_bin'] = (dfnew['sat_val'] > 0) * 1

# logit using TI and Q as predictors
model = smf.logit('sat_bin ~ TI_filtered + Q', data=dfnew).fit()

# check model performance
print(model.summary())

# predict in sample
in_sample = pd.DataFrame({'prob':model.predict()})

#%% make a sensitivity-specificity plot for model prediction

# thresholds = np.linspace(0.2,0.8,20)
thresholds = np.linspace(0.05,0.6,20)
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
ax.set_xlabel('1-specificity')
ax.set_ylabel('sensitivity')
fig.colorbar(sc, label='threshold')
# plt.savefig(save_directory+'sens_spec_DR.png')
plt.savefig(save_directory+'sens_spec_BR.png')

#%% predict odds of saturation in sample with the model

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


#%% get all the topographic index values for Druids Run

areafile = "LSDTT/baltimore2015_DR1_d8_area.bil"
af = rd.open(path+areafile)
area = af.read(1).astype(float)
af.close()

TIfile = "LSDTT/baltimore2015_DR1_TIfiltered.tif" # Druids Run
tif = rd.open(path+TIfile)
TI = tif.read(1).astype(float)
TI = TI[area>0] # np.ma.masked_array(TI, mask=area==-9999)

#%% Predict out of sample, and plot with TI CDF

TI1 = np.linspace(1,12,100)
Q_all = np.linspace(0.01,6,5)
fig, ax = plt.subplots()

for i, Q in enumerate(Q_all):
    pred = model.predict(exog=dict(TI_filtered=TI1, Q=Q*np.ones_like(TI1)))

    ax.plot(TI1, pred, color=cm.viridis(Q/max(Q_all)), label='Q=%.2f'%Q)
ax.axvspan(dfnew['TI_filtered'].min(), dfnew['TI_filtered'].max(), alpha=0.2, color='r')
ax.set_ylim(-0.05,1.05)
ax.legend(frameon=False)
ax.set_xlabel('TI')
ax.set_ylabel(r'Modeled P$(saturated)$')

ax1 = ax.twinx()
ax1.plot(np.sort(TI), np.linspace(0,1,len(TI)), color='k', linewidth=1, label='CDF')
ax1.set_xlim((1,12))
ax1.set_ylim(-0.05,1.05)
ax1.set_ylabel(r'P$(TI \leq TI_x)$')
ax1.legend(frameon=False, loc='lower right')
plt.savefig(save_directory+'pred_sat_ti_DR.png')


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
