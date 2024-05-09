
#%%

import numpy as np
import pandas as pd
import pickle
import glob
import rasterio as rd
import statsmodels.formula.api as smf
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker

save_directory = '/Users/dlitwin/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/'

#%% pick site files

site = 'DR'
res = 5 # resolution (m)

if site=='DR' and res>=1:
    path = '/Users/dlitwin/Documents/Research/Soldiers Delight/data/'
    processed_path = '/Users/dlitwin/Documents/Research/Soldiers Delight/data_processed/'
    TIfile = 'LSDTT/%d_meter/baltimore2015_DR1_%dm_TI.tif'%(res,res)
    curvfile = "LSDTT/%d_meter/baltimore2015_DR1_%dm_CURV.bil"%(res,res) # Druids Run
    basin_name = 'LSDTT/%d_meter/baltimore2015_DR1_%dm_AllBasins.bil'%(res,res)
    HSfile_res = 'LSDTT/%d_meter/baltimore2015_DR1_%dm_hs.bil'%(res,res)
    HSfile = 'LSDTT/baltimore2015_DR1_hs.bil' # Druids Run hillshade (full resolution)

elif site=='BR' and res>=1:
    path = '/Users/dlitwin/Documents/Research/Oregon Ridge/data/'
    processed_path = '/Users/dlitwin/Documents/Research/Oregon Ridge/data_processed/'
    TIfile = 'LSDTT/%d_meter/baltimore2015_BR_%dm_TI.tif'%(res,res)
    curvfile = "LSDTT/%d_meter/baltimore2015_BR_%dm_CURV.bil"%(res,res) # Baisman Run
    basin_name = 'LSDTT/%d_meter/baltimore2015_BR_%dm_AllBasins.bil'%(res,res)
    HSfile_res = 'LSDTT/%d_meter/baltimore2015_BR_%dm_hs.bil'%(res,res)
    HSfile = 'LSDTT/baltimore2015_BR_hs.bil' # Druids Run hillshade (full resolution)

elif site=='DR' and res<1:
    path = '/Users/dlitwin/Documents/Research/Soldiers Delight/data/'
    processed_path = '/Users/dlitwin/Documents/Research/Soldiers Delight/data_processed/'
    TIfile = 'LSDTT/baltimore2015_DR1_TIfiltered.tif'
    curvfile = "LSDTT/baltimore2015_DR1_CURV.bil" # Druids Run
    basin_name = 'LSDTT/baltimore2015_DR1_AllBasins.bil'
    HSfile = 'LSDTT/baltimore2015_DR1_hs.bil' # Druids Run hillshade (full resolution)    

elif site=='BR' and res<1:
    path = '/Users/dlitwin/Documents/Research/Oregon Ridge/data/'
    processed_path = '/Users/dlitwin/Documents/Research/Oregon Ridge/data_processed/'
    TIfile = 'LSDTT/baltimore2015_BR_TIfiltered.tif'
    curvfile = "LSDTT/baltimore2015_BR_CURV.bil" # Druids Run
    basin_name = 'LSDTT/baltimore2015_BR_AllBasins.bil'
    HSfile = 'LSDTT/baltimore2015_BR_hs.bil' # Druids Run hillshade (full resolution)    

else:
    print('%s at res %d is not there'%(site,res))


#%% assemble all saturation dataframes

dfs = []
paths = glob.glob(path + "saturation/transects_*.csv")
for pts_path in paths:

    # load and remove non-sat points
    df = pd.read_csv(pts_path) # sampled pts
    A = [True if X in ['N', 'Ys', 'Yp', 'Yf'] else False for X in df['Name']]
    df = df[A]

    # get date - time in df is timezone aware, so utc=true converts to the time in UTC
    datetime = pd.to_datetime(df.BeginTime, utc=True)
    df['date'] = datetime.dt.date
    df['datetime_start'] = datetime.iloc[0]
    
    # add discharge to df 

    dfs.append(df)
dfall = pd.concat(dfs, axis=0)

#%% get discharge

if site=='DR':

    # load continuous discharge (we'll use it to fill one day where a dilution gage was not done)
    area_DR = 107e4 #m2
    dfq_cont = pd.read_csv(processed_path+'DruidRun_discharge_15min_2022_3-2022_9.csv', 
                            parse_dates=[0],
                            )
    dfq_cont.set_index('Datetime', inplace=True)
    dfq_cont['Q m/d'] = dfq_cont['Q m3/s'] * 3600 * 24 * (1/area_DR) # sec/hr * hr/d * 1/m2

    # load dilution gaged Q for Druids Run
    q_DR = pickle.load(open(processed_path+'discharge_DR.p', 'rb'))
    dfq = pd.DataFrame.from_dict(q_DR, orient='index', dtype=None, columns=['Q']) # Q in L/s
    t1 = pd.Timestamp('2022-04-27 14:45:00')
    dfq.loc[t1] = dfq_cont['Q m3/s'].loc[t1] * 1000
    dfq = dfq.sort_index()
    dfq = dfq.tz_localize('America/New_York')
    dfq['datetime'] = dfq.index
    dfq['Q m/d'] = dfq['Q'] * (1/1000) * 3600 * 24 * (1/area_DR) # m3/liter * sec/hr * hr/d * 1/m2
    dfq['date'] = dfq['datetime'].dt.date

elif site=='BR':

    path_BR =path+"/USGS/Discharge_01583580_20220401.csv"
    path_PB =path+"/USGS/Discharge_01583570_20220401.csv"
    dfq_cont = pd.read_csv(path_BR, header=14)
    dfqug_cont = pd.read_csv(path_PB, header=14)

    area_BR = 381e4 #m2
    dfq_cont['Q m/d'] = dfq_cont['Value']*0.3048**3 * 3600 * 24 * (1/area_BR) #m3/ft3 * sec/hr * hr/d * 1/m2
    dfq_cont['datetime'] = pd.to_datetime(dfq_cont['ISO 8601 UTC'], utc=True)
    dfq_cont.set_index('datetime', inplace=True)
    dfq_cont = dfq_cont.filter(['Q m/d', 'Grade'])

    # get the start times of every saturation survey
    times = dfall.datetime_start.unique()
    times = [time.round('5min') for time in times]

    # isolate discharge at those times
    dfq = dfq_cont.loc[times]
    dfq['date'] = dfq.index.date

    area_PB = 37e4 #m2
    dfqug_cont['Q m/d'] = dfqug_cont['Value']*0.3048**3 * 3600 * 24 * (1/area_PB) #m3/ft3 * sec/hr * hr/d * 1/m2
    dfqug_cont['datetime'] = pd.to_datetime(dfqug_cont['ISO 8601 UTC'], utc=True)
    dfqug_cont.set_index('datetime', inplace=True)
    dfqug_cont = dfqug_cont.filter(['Q m/d', 'Grade'])

    dfqug = dfqug_cont.loc[times]
    dfqug['date'] = dfqug.index.date


#%% merged dataframe with saturation, discharge, and TI

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

# drop unnecessary columns, add binary classified saturation
dfnew.drop(columns=['OID_', 'BeginTime', 'Unnamed: 0', 'FolderPath'], inplace=True, errors='ignore')
dfnew['sat_bin'] = (dfnew['sat_val'] > 0) * 1

# save all of this (used for transmissivity estimation)
dfnew.to_csv(processed_path + f"saturation_{site}_{res}.csv")

#%% Get TI of the basin

# get basin
bsn = rd.open(path + basin_name)
basin = bsn.read(1) > 0 
bounds = bsn.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]

# get all TI
tif = rd.open(path+TIfile)
TI = tif.read(1).astype(float)
TI = TI[basin]

plt.figure()
plt.imshow(basin,
            origin="upper", 
            extent=Extent,
            cmap='viridis',
            )
plt.show()


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
plt.savefig(save_directory+f'sat_{site}_{res}m.pdf', transparent=True)
plt.savefig(save_directory+f'sat_{site}_{res}m.png', transparent=True)
plt.show()
#%% Saturation on hillshade combined with saturation-TI

cols = {'N':"peru", 'Ys':"dodgerblue", 'Yp':"blue", 'Yf':"navy"}

src = rd.open(path + HSfile) # hillshade
bounds = src.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
Extent_90 = [bounds.bottom,bounds.top,bounds.right,bounds.left]

fig, axs = plt.subplots(nrows=len(dfnew.date.unique()), ncols=2, figsize=(5,6)) #(5,8)

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
    axs[i,0].set_xlabel(r'$Q = %.2f$ mm/d'%(df['Q m/d'].iloc[0]*1000))  
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

fig, axs = plt.subplots(nrows=len(dfnew.date.unique()), figsize=(3,8)) #(3,6)

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
    axs[i].set_xlabel(r'$Q = %.2f$ mm/d'%(df['Q m/d'].iloc[0]*1000))  
    # axs[i].set_xlabel('')
    axs[i].set_ylabel('')
    axs[i].set_title(str(date), fontsize=10)

    i += 1

plt.tight_layout()
plt.savefig(save_directory+f'sat_{site}_{res}m_rows.pdf', transparent=True)
plt.savefig(save_directory+f'sat_{site}_{res}m_rows.png', transparent=True, dpi=300)
plt.show()

#%% Logistic regression: sat_bin ~ TI_filtered + logQ

# make saturation into a binary field
dfnew['logQ'] = np.log(dfnew['Q m/d'])

# logit using TI and Q as predictors
model = smf.logit('sat_bin ~ TI_filtered + logQ', data=dfnew).fit()

# check model performance
print(model.summary())
with open(save_directory+f'summary_{site}_logTI_logQ_{res}.txt', 'w') as fh:
    fh.write(model.summary().as_text())

# predict in sample
in_sample = pd.DataFrame({'prob':model.predict()})

#%% Predict out of sample, and plot with TI PDF

TI1 = np.linspace(0.01,22,100)
Q_all = np.geomspace(dfnew['Q m/d'].min(),dfnew['Q m/d'].max(),5)
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

p_best = 0.5 # set p by hand

TI_plot = tif.read(1).astype(float)

Q_all = dfq_cont['Q m/d'].values
TI_range = np.linspace(np.min(TI_plot), np.max(TI_plot), 500)
sat_state = np.zeros_like(TI_range)


Q_all = Q_all[~np.isnan(Q_all)]
for i, Q in enumerate(Q_all):
    pred = model.predict(exog=dict(TI_filtered=TI_range, logQ=np.log(Q)*np.ones_like(TI_range)))
    sat_state += 1*(pred.values>p_best)

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
dfsat.to_csv('/Users/dlitwin/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/df_sat_%s.csv'%site)


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

figsize = (9,5) if site=='BR' else (5,5)
fig, ax = plt.subplots(figsize=figsize) 
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

# %%
