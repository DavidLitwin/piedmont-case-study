
#%%

import numpy as np
import pandas as pd
import pickle
import glob
import rasterio as rd
import statsmodels.formula.api as smf
from sklearn.neighbors import KernelDensity
import dataretrieval.nwis as nwis

import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker

save_directory = 'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/'

#%%

site = 'BR'
res = 5

if site=='DR' and res>=1:
    path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/'
    TIfile = 'LSDTT/%d_meter/baltimore2015_DR1_%dm_TI.tif'%(res,res)
    curvfile = "LSDTT/%d_meter/baltimore2015_DR1_%dm_CURV.bil"%(res,res) # Druids Run
    basin_name = 'LSDTT/%d_meter/baltimore2015_DR1_%dm_AllBasins.bil'%(res,res)
    HSfile_res = 'LSDTT/%d_meter/baltimore2015_DR1_%dm_hs.bil'%(res,res)
    HSfile = 'LSDTT/baltimore2015_DR1_hs.bil' # Druids Run hillshade (full resolution)

elif site=='BR' and res>=1:
    path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/'
    TIfile = 'LSDTT/%d_meter/baltimore2015_BR_%dm_TI.tif'%(res,res)
    curvfile = "LSDTT/%d_meter/baltimore2015_BR_%dm_CURV.bil"%(res,res) # Baisman Run
    basin_name = 'LSDTT/%d_meter/baltimore2015_BR_%dm_AllBasins.bil'%(res,res)
    HSfile_res = 'LSDTT/%d_meter/baltimore2015_BR_%dm_hs.bil'%(res,res)
    HSfile = 'LSDTT/baltimore2015_BR_hs.bil' # Druids Run hillshade (full resolution)

elif site=='DR' and res<1:
    path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/'
    TIfile = 'LSDTT/baltimore2015_DR1_TIfiltered.tif'
    curvfile = "LSDTT/baltimore2015_DR1_CURV.bil" # Druids Run
    basin_name = 'LSDTT/baltimore2015_DR1_AllBasins.bil'
    HSfile = 'LSDTT/baltimore2015_DR1_hs.bil' # Druids Run hillshade (full resolution)    

elif site=='BR' and res<1:
    path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/'
    TIfile = 'LSDTT/baltimore2015_BR_TIfiltered.tif'
    curvfile = "LSDTT/baltimore2015_BR_CURV.bil" # Druids Run
    basin_name = 'LSDTT/baltimore2015_BR_AllBasins.bil'
    HSfile = 'LSDTT/baltimore2015_BR_hs.bil' # Druids Run hillshade (full resolution)    

else:
    print('%s at res %d is not there'%(site,res))

#%% individual saturation survey

sat_path = path + "saturation/transects_20230126.csv"

cols = {'N':"peru", 'Ys':"dodgerblue", 'Yp':"blue", 'Yf':"navy"}

fig, ax = plt.subplots(figsize=(4,4))

src = rd.open(path + HSfile) # hillshade
df = pd.read_csv(sat_path) # sampled pts
A = [True if X in ['N', 'Ys', 'Yp', 'Yf'] else False for X in df['Name']]
df = df[A]

bounds = src.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
Extent_90 = [bounds.bottom,bounds.top,bounds.right,bounds.left]

grouped = df.groupby('Name')
for key, group in grouped:
    if site=='DR':
        group.plot(ax=ax, kind='scatter', x='X', y='Y', 
                    label=key, color=cols[key], legend=False, s=20
                    )
        cs = ax.imshow(src.read(1), cmap='binary', 
                        extent=Extent, vmin=100, origin="upper")
        ax.set_xlim((341200, 341500)) # DR bounds
        ax.set_ylim((4.36490e6,4.36511e6)) # DR Bounds
    else:
        group.plot(ax=ax, kind='scatter', x='Y', y='X', 
                    label=key, color=cols[key], legend=False, s=20
                    )
        cs = ax.imshow(np.rot90(src.read(1), k=3),
                        cmap='binary', 
                        extent=Extent_90, 
                        vmin=100,
                        origin="upper")       
        ax.set_ylim((355100,354600)) # PB bounds
        ax.set_xlim((4.3715e6,4.3722e6)) # PB Bounds
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')
# ax.set_title(df.BeginTime[1][0:10])

plt.tight_layout()
# plt.savefig(save_directory+f'sat_{site}_{res}m.pdf', transparent=True)
# plt.savefig(save_directory+f'sat_{site}_{res}m.png', transparent=True)
plt.show()


#%% assemble all saturation dataframes

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
dfq_cont = pd.read_csv(q_path+'DruidRun_discharge_15min_2022_3-2022_9.csv', 
                        parse_dates=[0],
                        infer_datetime_format=True)
dfq_cont.set_index('Datetime', inplace=True)
dfq_cont['Q m/d'] = dfq_cont['Q m3/s'] * 3600 * 24 * (1/area_DR) # sec/hr * hr/d * 1/m2

# load dilution gaged Q for Druids Run
q_DR = pickle.load(open(q_path+'discharge_DR.p', 'rb'))
dfq = pd.DataFrame.from_dict(q_DR, orient='index', dtype=None, columns=['Q']) # Q in L/s
t1 = pd.Timestamp('2022-04-27 14:45:00')
dfq.loc[t1] = dfq_cont['Q m3/s'].loc[t1] * 1000
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

dfq_cont = nwis.get_record(sites=site_BR, service='iv', start='2022-08-01', end='2023-03-21')
dfqug_cont = nwis.get_record(sites=site_PB, service='iv', start='2022-08-01', end='2023-03-21')

# load existing data
# path_BR = "C:/Users/dgbli/Documents/Research/Oregon Ridge/data_processed/"
# dfq_cont = pd.read_csv(path_BR+'dfq.csv', index_col='datetime')
# dfqug_cont = pd.read_csv(path_BR+'dfqug.csv', index_col='datetime')

#%% Baisman run: process Q

# area normalized discharge
area_BR = 381e4 #m2
dfq_cont['Q m/d'] = dfq_cont['00060']*0.3048**3 * 3600 * 24 * (1/area_BR) #m3/ft3 * sec/hr * hr/d * 1/m2
dfq_cont.drop(columns=['00060', 'site_no', '00065', '00065_cd'], inplace=True)

area_PB = 37e4 #m2
dfqug_cont['Q m/d'] = dfqug_cont['00060']*0.3048**3 * 3600 * 24 * (1/area_PB) #m3/ft3 * sec/hr * hr/d * 1/m2
dfqug_cont.drop(columns=['00060', 'site_no', '00065', '00065_cd'], inplace=True)

# index from string to datetime
dfq_cont['datetime'] = pd.to_datetime(dfq_cont.index, utc=True)
dfq_cont.set_index('datetime', inplace=True)

dfqug_cont['datetime'] = pd.to_datetime(dfqug_cont.index, utc=True)
dfqug_cont.set_index('datetime', inplace=True)

# get the start times of every saturation survey
times = dfall.datetime_start.unique()
times = [time.round('5min') for time in times]

# isolate discharge at those times
dfq = dfq_cont.loc[times]
dfq['datetime'] = dfq.index
dfq['date'] = dfq['datetime'].dt.date

dfqug = dfqug_cont.loc[times]
dfqug['datetime'] = dfqug.index
dfqug['date'] = dfqug['datetime'].dt.date

#%% Add to saturation dataframe, get all TI

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
dfnew['sat_bin'] = (dfnew['sat_val'] > 0) * 1

# get basin
bsn = rd.open(path + basin_name)
basin = bsn.read(1) > 0 
bounds = bsn.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]

plt.figure()
plt.imshow(basin,
            origin="upper", 
            extent=Extent,
            cmap='viridis',
            )
plt.show()

# get all TI
tif = rd.open(path+TIfile)
TI = tif.read(1).astype(float)
TI = TI[basin]


#%% Saturation on hillshade combined with saturation-TI

cols = {'N':"peru", 'Ys':"dodgerblue", 'Yp':"blue", 'Yf':"navy"}

src = rd.open(path + HSfile) # hillshade
bounds = src.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
Extent_90 = [bounds.bottom,bounds.top,bounds.right,bounds.left]

fig, axs = plt.subplots(nrows=len(dfnew.date.unique()), ncols=2, figsize=(5,8)) #(5,6)

grouped_date = dfnew.groupby('date')
i = 0
for date, df in grouped_date:

    grouped = df.groupby('Name')
    for key, group in grouped:
        if site=='DR':
            group.plot(ax=axs[i,0], kind='scatter', x='X', y='Y', 
                        label=key, color=cols[key], legend=False, s=10
                        )
            cs = axs[i,0].imshow(src.read(1), cmap='binary', 
                            extent=Extent, vmin=100, origin="upper")
            axs[i,0].set_xlim((341200, 341500)) # DR bounds
            axs[i,0].set_ylim((4.36490e6,4.36511e6)) # DR Bounds

            if i==0:
                axs[i,0].plot([341430, 341480], [4364920, 4364920], color='k', linewidth=4)
                axs[i,0].text(341425, 4364930, '50 m', fontsize=8)
        else:
            group.plot(ax=axs[i,0], kind='scatter', x='Y', y='X', 
                        label=key, color=cols[key], legend=False, s=10
                        )
            cs = axs[i,0].imshow(np.rot90(src.read(1), k=3),
                            cmap='binary', 
                            extent=Extent_90, 
                            vmin=100,
                            origin="upper")   
            axs[i,0].set_ylim((355100,354600)) # PB bounds
            axs[i,0].set_xlim((4.3715e6,4.3722e6)) # PB Bounds

    axs[i,0].set_xticks([])
    axs[i,0].set_yticks([])
    axs[i,0].set_xlabel(r'$Q = %.2f$ mm/d'%(df['Q'].iloc[0]*1000))  
    # axs[i,0].set_xlabel('')
    axs[i,0].set_ylabel('')
    axs[i,0].set_title(str(date), fontsize=10)

    np.random.seed(2023)
    df['sat_perturbed'] = df['sat_val'] + 0.05*np.random.randn(len(df))

    for key, group in grouped:
            group.plot(ax=axs[i,1], kind='scatter', x='TI_filtered', y='sat_perturbed', 
                        label=key, color=cols[key], legend=False,
                        )
    axs[i,1].set_yticks([0,1,2,3])
    axs[i,1].set_yticklabels(['N', 'Ys', 'Yp', 'Yf'])
    axs[i,1].set_ylabel('')

    if i == len(paths)-1:
        axs[i,1].set_xlabel('$\log(TI)$')
    else:
        axs[i,1].set_xlabel('')

    i += 1

plt.tight_layout()
plt.savefig(save_directory+f'sat_TI_{site}_{res}m.pdf', transparent=True)
plt.savefig(save_directory+f'sat_TI_{site}_{res}m.png', transparent=True)
plt.show()

#%% just map view

cols = {'N':"peru", 'Ys':"dodgerblue", 'Yp':"blue", 'Yf':"navy"}

src = rd.open(path + HSfile) # hillshade
bounds = src.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
Extent_90 = [bounds.bottom,bounds.top,bounds.right,bounds.left]

fig, axs = plt.subplots(nrows=len(dfnew.date.unique()), figsize=(3,6)) #(3,8)

grouped_date = dfnew.groupby('date')
i = 0
for date, df in grouped_date:

    grouped = df.groupby('Name')
    for key, group in grouped:
        if site=='DR':
            group.plot(ax=axs[i], kind='scatter', x='X', y='Y', 
                        label=key, color=cols[key], legend=False, s=10
                        )
            cs = axs[i].imshow(src.read(1), cmap='binary', 
                            extent=Extent, vmin=100, origin="upper")
            axs[i].set_xlim((341200, 341500)) # DR bounds
            axs[i].set_ylim((4.36490e6,4.36511e6)) # DR Bounds

            if i==0:
                axs[i].plot([341430, 341480], [4364920, 4364920], color='k', linewidth=4)
                axs[i].text(341425, 4364930, '50 m', fontsize=8)
        else:
            group.plot(ax=axs[i], kind='scatter', x='Y', y='X', 
                        label=key, color=cols[key], legend=False, s=10
                        )
            cs = axs[i].imshow(np.rot90(src.read(1), k=3),
                            cmap='binary', 
                            extent=Extent_90, 
                            vmin=100,
                            origin="upper")   
            if i==0:
                axs[i].plot([4372000, 4372100], [355050, 355050], color='k', linewidth=4)
                axs[i].text(4371980, 355020, '100 m', fontsize=8)

            axs[i].set_ylim((355100,354600)) # PB bounds
            axs[i].set_xlim((4.3715e6,4.3722e6)) # PB Bounds

    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_xlabel(r'$Q = %.2f$ mm/d'%(df['Q'].iloc[0]*1000))  
    # axs[i].set_xlabel('')
    axs[i].set_ylabel('')
    axs[i].set_title(str(date), fontsize=10)

    i += 1

plt.tight_layout()
plt.savefig(save_directory+f'sat_{site}_{res}m_rows.pdf', transparent=True)
plt.savefig(save_directory+f'sat_{site}_{res}m_rows.png', transparent=True, dpi=300)
plt.show()

#%% location on hydrograph

if site == 'DR':
    path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data_processed/'
else:
    path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data_processed/'
dfq_plot = pd.read_csv(path+'df_qbp.csv', index_col='Date', parse_dates=True)


P_plot = dfq_plot['P (mm)'] * 4 # mm/hr
Q_plot = dfq_plot['Total runoff [m^3 s^-1]'] * (1/area_DR) * 3600 * 1000 # 1/m2 sec/hr mm/m
Qb_plot = dfq_plot['Baseflow [m^3 s^-1]'] * (1/area_DR) * 3600 * 1000 # 1/m2 sec/hr mm/m

times = dfnew.datetime_start.unique()

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(Q_plot, 'k-')
ax.plot(Qb_plot, 'b-')
ax.scatter(times, 
           np.ones(len(times)), 
           color='r', s=10, zorder=101)
# ax.set_xlim((date(2022,4,1), date(2022,12,1)))
ax.set_yscale('log')
ax.set_ylabel('$Q$, $Q_b$ (mm/hr)')

#%% Logistic regression: sat_bin ~ logTIQ

# logTIQ = TI + log(Q) because TI is already log
dfnew['logTIQ'] = dfnew['TI_filtered'] + np.log(dfnew['Q'])

# logit using TI and Q as predictors
model = smf.logit('sat_bin ~ logTIQ', data=dfnew).fit()

# check model performance
print(model.summary())
with open(save_directory+f'summary_{site}_logTIQ_{res}.txt', 'w') as fh:
    fh.write(model.summary().as_text())

# predict in sample
in_sample = pd.DataFrame({'prob':model.predict()})

#%% calculate transmissivity: range of thresholds

N = 1000
# p_all = np.linspace(0.2,0.7,N) #DR
p_all = np.linspace(0.05,0.7,N) #BR
rhostar = lambda p: np.log(p/(1-p))
Tmean = lambda b0, b1, p: np.exp((rhostar(p)-b0)/b1)

T_all = np.zeros((N,3))
sens = np.zeros(N)
spec = np.zeros(N)
dist = np.zeros(N)

covs = model.cov_params()
means = model.params

samples = np.random.multivariate_normal(means, covs, size=100000)

for i, p in enumerate(p_all):

    Tcalc = Tmean(samples[:,0], samples[:,1], p)
    T_all[i,0] = np.median(Tcalc)
    T_all[i,1] = np.percentile(Tcalc, 25)
    T_all[i,2] = np.percentile(Tcalc, 75)

    in_sample['pred_label'] = (in_sample['prob']>p).astype(int)
    cs = pd.crosstab(in_sample['pred_label'],dfnew['sat_bin'])
    sens[i] = cs[1][1]/(cs[1][1] + cs[1][0])
    spec[i] = cs[0][0]/(cs[0][0] + cs[0][1])

    dist[i] = abs(sens[i]-(1-spec[i]))

#%%

s1 = 8
s2 = (5,4)

fig, ax = plt.subplots(figsize=s2)
sc = ax.scatter(1-spec, sens, c=p_all, s=s1, vmin=0.0, vmax=0.7)
ax.axline([0,0], [1,1], color='k', linestyle='--', label='1:1')
ax.set_xlabel('False Positive Ratio (FPR)')
ax.set_ylabel('True Positive Ratio (TPR)')
fig.colorbar(sc, label=r'$p^*$')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig(save_directory+'FPR_TPR_thresh_%s.png'%site, dpi=300, transparent=True)
plt.savefig(save_directory+'FPR_TPR_thresh_%s.pdf'%site, transparent=True)

fig, ax = plt.subplots(figsize=s2)
sc = ax.scatter(1-spec, sens, c=T_all[:,0], s=s1, cmap='plasma', norm=colors.LogNorm())
ax.axline([0,0], [1,1], color='k', linestyle='--', label='1:1')
ax.set_xlabel('False Positive Ratio (FPR)')
ax.set_ylabel('True Positive Ratio (TPR)')
fig.colorbar(sc, label='Transmissivity')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig(save_directory+'FPR_TPR_trans_%s.png'%site, dpi=300, transparent=True)
plt.savefig(save_directory+'FPR_TPR_trans_%s.pdf'%site, transparent=True)

fig, ax = plt.subplots(figsize=s2)
sc = ax.scatter(T_all[:,0], dist, c=p_all, s=s1, vmin=0.0, vmax=0.7)
# ax.plot(T_all[:,0], dist, 'k', linewidth=0.5, zorder=100)
ax.set_xscale('log')
ax.set_xlabel('Transmissivity')
ax.set_ylabel('TPR-FPR')
fig.colorbar(sc, label=r'$p^*$')
fig.tight_layout()
plt.savefig(save_directory+'trans_dist_thresh_%s.png'%site, dpi=300, transparent=True)
plt.savefig(save_directory+'trans_dist_thresh_%s.pdf'%site, transparent=True)

#%% select max and save

i = np.argmax(dist)
T_select = T_all[i,:]

dfT = pd.DataFrame({'Trans. med [m2/d]':T_select[0],
              'Trans. lq [m2/d]': T_select[1],
              'Trans. uq [m2/d]': T_select[2],
              'b0':means[0],
              'b1':means[1],
              'thresh':p_all[i],
              'rho':rhostar(p_all[i]),
              'dist':dist[i]},
              index=[0]
              )
dfT.to_csv(save_directory+f'transmissivity_{site}_logTIQ_{res}.csv', float_format='%.3f')


#%% Predict out of sample, and plot with TI CDF

TI1 = np.linspace(0.01,22,100)
Q_all = np.geomspace(dfnew['Q'].min(),dfnew['Q'].max(),5)
# Q_all = np.linspace(0.02,0.5,5)
fig, ax = plt.subplots()

for i, Q in enumerate(Q_all):
    pred = model.predict(exog=dict(logTIQ=TI1 + np.log(Q)))

    ax.plot(TI1, pred, color=cm.viridis(Q/max(Q_all)), label='Q=%.2f mm/d'%(Q*1000))
ax.axvspan(dfnew['TI_filtered'].min(), dfnew['TI_filtered'].max(), alpha=0.2, color='r')
ax.set_ylim(-0.05,1.05)
ax.set_xlim((0.0,22))
ax.legend(frameon=False, loc='upper left')
ax.set_xlabel('TI')
ax.set_ylabel(r'Modeled P$(saturated)$')

ax1 = ax.twinx()
ax1.plot(np.sort(TI), np.linspace(0,1,len(TI)), color='k', linewidth=1, label='CDF')
ax1.set_xlim((0.0,22))
ax1.set_ylim(-0.05,1.05)
ax1.set_ylabel(r'P$(TI \leq TI_x)$')
ax1.legend(frameon=False, loc='lower right')
plt.savefig(save_directory+f'pred_sat_ti_{site}_logTIQ_{res}.png')

#%% Predict out of sample, and plot with TI PDF

TI1 = np.linspace(0.01,22,500)
Q_all = np.geomspace(dfnew['Q'].min(),dfnew['Q'].max(),5)
# Q_all = np.linspace(0.02,0.5,5)
fig, ax = plt.subplots()

for i, Q in enumerate(Q_all):
    pred = model.predict(exog=dict(logTIQ=TI1 + np.log(Q)))

    ax.plot(TI1, pred, color=cm.viridis(Q/max(Q_all)), label='Q=%.2f mm/d'%(Q*1000))
ax.axvspan(dfnew['TI_filtered'].min(), dfnew['TI_filtered'].max(), alpha=0.2, color='r')
ax.set_ylim(-0.01,1.05)
ax.set_xlim((0.0,22))
ax.legend(frameon=False, loc='upper left')
ax.set_xlabel('TI')
ax.set_ylabel(r'Modeled P$(saturated)$')

ax1 = ax.twinx()
kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
kde.fit(TI.reshape(-1,1))
logprob = kde.score_samples(TI1.reshape(-1,1))
ax1.fill_between(TI1, np.exp(logprob), alpha=0.5)
ax1.set_xlim((0.0,22))
ax1.set_ylim(-0.01,0.5)
ax1.set_ylabel('Density')
ax1.legend(frameon=False, loc='lower right')
# plt.savefig(save_directory+f'pred_sat_ti_{site}_logTIQ_{res}.png')

#%% Logistic regression: sat_bin ~ TI_filtered + logQ

# make saturation into a binary field
dfnew['logQ'] = np.log(dfnew['Q'])

# logit using TI and Q as predictors
model = smf.logit('sat_bin ~ TI_filtered + logQ', data=dfnew).fit()

# check model performance
print(model.summary())
with open(save_directory+f'summary_{site}_logTI_logQ_{res}.txt', 'w') as fh:
    fh.write(model.summary().as_text())

# predict in sample
in_sample = pd.DataFrame({'prob':model.predict()})

#%% Predict out of sample, and plot with TI CDF

TI1 = np.linspace(0.01,22,100)
Q_all = np.geomspace(dfnew['Q'].min(),dfnew['Q'].max(),5)
fig, ax = plt.subplots()

for i, Q in enumerate(Q_all):
    pred = model.predict(exog=dict(TI_filtered=TI1, logQ=np.log(Q)*np.ones_like(TI1)))

    ax.plot(TI1, pred, color=cm.viridis(Q/max(Q_all)), label='Q=%.2f mm/d'%(Q*1000))
ax.axvspan(dfnew['TI_filtered'].min(), dfnew['TI_filtered'].max(), alpha=0.2, color='r')
ax.set_ylim(-0.05,1.05)
ax.set_xlim((0.0,12))
ax.legend(frameon=False, loc='upper left')
ax.set_xlabel('TI')
ax.set_ylabel(r'Modeled P$(saturated)$')

ax1 = ax.twinx()
ax1.plot(np.sort(TI), np.linspace(0,1,len(TI)), color='k', linewidth=1, label='CDF')
ax1.set_xlim((0.0,12))
ax1.set_ylim(-0.05,1.05)
ax1.set_ylabel(r'P$(TI \leq TI_x)$')
ax1.legend(frameon=False, loc='lower right')
plt.savefig(save_directory+f'pred_sat_ti_cdf_{site}_logTI_logQ_{res}.pdf', transparent=True)

#%% Predict out of sample, and plot with TI PDF

TI1 = np.linspace(0.01,22,100)
Q_all = np.geomspace(dfnew['Q'].min(),dfnew['Q'].max(),5)
fig, ax = plt.subplots(figsize=(6,4))

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
for i, Q in enumerate(Q_all):
    pred = model.predict(exog=dict(TI_filtered=TI1, logQ=np.log(Q)*np.ones_like(TI1)))
    ax.plot(TI1, pred, color=cm.viridis(Q/max(Q_all)), label='Q=%.2f mm/d'%(Q*1000))
    
    TIc = (-model.params[0]-model.params[2]*np.log(Q))/model.params[1]
    ax.plot([TIc,TIc], [0.0,0.5], color=cm.viridis(Q/max(Q_all)), linestyle=':', linewidth=1)
ax.set_yticks([0.0,0.25,0.5,0.75,1.0])
ax.set_ylim(-0.01,1.01)
ax.set_xlim((0.0,22))
ax.legend(frameon=False, loc='upper left')
ax.set_xlabel('log($TI$)')
ax.set_ylabel(r'Modeled P$(saturated)$')

ax1 = ax.twinx()
kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
kde.fit(TI.reshape(-1,1))
logprob = kde.score_samples(TI1.reshape(-1,1))
ax1.axvspan(dfnew['TI_filtered'].min(), dfnew['TI_filtered'].max(), alpha=0.1, color='peru')
ax1.fill_between(TI1, np.exp(logprob), alpha=0.4, color='peru')
ax1.plot(TI1, np.exp(logprob), color='peru', linewidth=1, label='PDF')
ax1.set_xlim((0.0,22))
ax1.set_ylim(-0.01,0.7)
ax1.set_ylabel('Density')
ax1.legend(frameon=False, loc='lower right')
fig.tight_layout()
plt.savefig(save_directory+f'pred_sat_ti_pdf_{site}_logTI_logQ_{res}.pdf', transparent=True)
plt.savefig(save_directory+f'pred_sat_ti_pdf_{site}_logTI_logQ_{res}.png', transparent=True)

#%% derive sat class from regression

TI_plot = tif.read(1).astype(float)

Q_all = dfq_cont['Q m/d'].values
TI_range = np.linspace(np.min(TI_plot), np.max(TI_plot), 500)
sat_state = np.zeros_like(TI_range)


Q_all = Q_all[~np.isnan(Q_all)]
for i, Q in enumerate(Q_all):
    pred = model.predict(exog=dict(TI_filtered=TI_range, logQ=np.log(Q)*np.ones_like(TI_range)))
    sat_state += 1*(pred.values>0.5)

sat_state = sat_state / len(Q_all)
sat_class = np.ones_like(sat_state)
sat_class[sat_state < 0.05] = 0
sat_class[sat_state > 0.95] = 2

x = np.digitize(TI_plot, TI_range)
x[x==500] = 499
sat_class_plot = sat_class[x]

#%%
vals, counts = np.unique(sat_class_plot[basin], return_counts=True)

dfsat = pd.DataFrame(data=counts/len(sat_class_plot[basin]), 
                    index=['sat_never','sat_variable', 'sat_always'])
dfsat.to_csv('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/df_sat_%s.csv'%site)


#%% plot classified saturated areas

# colorbar approach courtesy of https://stackoverflow.com/a/53361072/11627361, https://stackoverflow.com/a/60870122/11627361, 

labels = ["dry", "variable", "wet"]    
L1 = ["peru", "dodgerblue", "navy"]

cmap = colors.ListedColormap(L1)
norm = colors.BoundaryNorm(np.arange(-0.5, 3), cmap.N)
fmt = ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

# hillshade
src = rd.open(path + HSfile_res) # hillshade
bounds = src.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
hs = src.read(1)
shp = TI_plot.shape
hs = np.ma.masked_array(hs, mask=hs==-9999)

fig, ax = plt.subplots(figsize=(5,5)) #(9,5)
ax.imshow(hs, cmap='binary', 
                extent=Extent, origin="upper", vmin=160)
im = ax.imshow(sat_class_plot.reshape(shp), 
                        origin="upper", 
                        extent=Extent, 
                        cmap=cmap,
                        norm=norm,
                        alpha=0.7,
                        interpolation="none",
                        )
if site=='DR':
    ax.plot([341400, 341650], [4364700, 4364700], color='k', linewidth=4)
    ax.text(341430, 4364730, '250 m', fontsize=10)
if site=='BR':
    ax.plot([355000, 355500], [4370400, 4370400], color='k', linewidth=4)
    ax.text(355100, 4370440, '500 m', fontsize=10)

fig.colorbar(im, format=fmt, ticks=np.arange(0,3))
ofs= 100 #50
ax.set_xlim((Extent[0]+ofs, Extent[1]-ofs))
ax.set_ylim((Extent[2]+ofs,Extent[3]-ofs))
plt.axis('off')
plt.savefig(save_directory+f'satclass_{site}_{res}.pdf', transparent=True)
plt.savefig(save_directory+f'satclass_{site}_{res}.png', transparent=True)

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
thresholds = np.linspace(0.2,0.8,20)
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
# plt.savefig(save_directory+'sens_spec_%s.png'%site)

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



#%% calulate a transmissivity: logTIQ method

p = 0.35
rhostar = np.log(p/(1-p))
Tmean = lambda b0, b1: np.exp((rhostar-b0)/b1)

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
# dfT.to_csv(save_directory+f'transmissivity_{site}_logTIQ_{res}.csv')


#%% logistic regression for each day separately

Tmean = lambda b0, b1, Q: np.exp((-b0)/b1)*Q


grouped = dfnew.groupby('date')
T_estimates = []
for key, group in grouped:

    model = smf.logit('sat_bin ~ TI_filtered', data=group).fit()

    covs = model.cov_params()
    means = model.params

    samples = np.random.multivariate_normal(means, covs, size=100000)

    Tcalc = Tmean(samples[:,0], samples[:,1], group['Q m/d'].iloc[0])
    T_median = np.median(Tcalc)
    T_lq = np.percentile(Tcalc, 25)
    T_uq = np.percentile(Tcalc, 75)

    T_estimates.append([T_median, T_lq, T_uq])


#%% plot saturation on hillshade

# hillshade
src = rd.open(path + HSfile_res) # hillshade
bounds = src.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]

# TI
TI_plot = tif.read(1).astype(float)
shp = TI_plot.shape
TI_plot = TI_plot.flatten()
sat_state = np.zeros_like(TI_plot)

Q_all = np.geomspace(dfnew['Q'].min(),dfnew['Q'].max(),5)
for i, Q in enumerate(Q_all):
    pred = model.predict(exog=dict(TI_filtered=TI_plot, logQ=np.log(Q)*np.ones_like(TI_plot)))
    sat_state += 1*(pred>0.5)

#%%

# sat_state = np.ma.masked_array(sat_state, mask=~basin)
hs = src.read(1)
hs = np.ma.masked_array(hs, mask=hs==-9999)

fig, ax = plt.subplots(figsize=(10,6)) #(6,6)
ax.imshow(hs, cmap='binary', 
                extent=Extent, origin="upper", vmin=160)
cmap = cm.get_cmap('Blues', 5)
cmap.set_under('w')
cs = ax.imshow(sat_state.values.reshape(shp), cmap=cmap, vmin=0.5,vmax=5.5, 
                extent=Extent, 
                origin="upper",
                alpha=0.7,
                interpolation=None,
                )
cbar = fig.colorbar(cs, ticks=[1,2,3,4,5], label='Q (mm/d)', extend='min')
label = np.round(Q_all[::-1]*1000,2)
cbar.ax.set_yticklabels(label) 
ofs= 100 #50
ax.set_xlim((Extent[0]+ofs, Extent[1]-ofs))
ax.set_ylim((Extent[2]+ofs,Extent[3]-ofs))
plt.axis('off')
# plt.savefig(save_directory+f'satmap_{site}_{res}.pdf', transparent=True)
# plt.savefig(save_directory+f'satmap_{site}_{res}.png', transparent=True)

# ax.set_xlim((341200, 341500)) # DR bounds
# ax.set_ylim((4.36490e6,4.36511e6)) # DR Bounds
# ax.set_xlim((355100,354600)) # PB bounds
# ax.set_ylim((4.3715e6,4.3722e6)) # PB Bounds

