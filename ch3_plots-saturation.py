
#%%

import numpy as np
import pandas as pd
import pickle
import glob
import rasterio as rd
from sklearn.linear_model import LogisticRegression
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors



# %% paths

save_directory = 'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/'

## Soldiers Delight:
path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/'
slopefile = "LSDTT/baltimore2015_DR1_SLOPE.bil"
areafile = "LSDTT/baltimore2015_DR1_d8_area.bil"

## Oregon Ridge
# path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/'
# slopefile = "LSDTT/baltimore2015_BR_SLOPE.bil"
# areafile = "LSDTT/baltimore2015_BR_d8_area.bil"

#%% import for topographic index

# fitted (10 m) slope
sf = rd.open(path+slopefile)
slope = sf.read(1).astype(float)
slope = np.ma.masked_array(slope, mask=slope==-9999)

dx = sf.transform[0]
bounds = sf.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
# sf.close()

# d8 area
af = rd.open(path+areafile)
area = af.read(1).astype(float)
area = np.ma.masked_array(area, mask=area==-9999)
af.close()

#%%
# calculate and filter topographic index 

TI = np.log(area/(slope * dx))

plt.figure()
plt.imshow(TI,
            origin="upper", 
            extent=Extent,
            cmap='viridis',
            )
plt.show()

TI_filtered = gaussian_filter(TI, sigma=4)

plt.figure()
plt.imshow(TI_filtered,
            origin="upper", 
            extent=Extent,
            cmap='viridis',
            )
plt.show()

# write TI filtered to .tif
TIfile = "LSDTT/baltimore2015_DR1_TIfiltered.tif" # Druids Run
# TIfile = "LSDTT/baltimore2015_BR_TIfiltered.tif" # Baisman Run
sf = rd.open(path+slopefile)
TI_dataset = rd.open(
    path+TIfile,
    'w',
    driver='GTiff',
    height=sf.height,
    width=sf.width,
    count=1,
    dtype=TI_filtered.dtype,
    crs=sf.crs,
    transform=sf.transform,
)
TI_dataset.write(TI_filtered,1)
TI_dataset.close()

#%% clean Pond Branch and Soldiers Delight EMLID REACH saturation files

path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/saturation/'
paths = glob.glob(path + "PB_UTM_*.csv")

for pth in paths:
    df = pd.read_csv(pth)
    df = df[['Name', 'Easting', 'Northing', 'Averaging start', 'Easting RMS', 'Northing RMS']]
    A = [True if X in ['N', 'Ys', 'Yp', 'Yf'] else False for X in df['Name']]
    df = df[A]

    namesdict = {'Easting':'X',
                'Northing': 'Y',
                'Averaging start': 'BeginTime',
                'Easting RMS': 'X_rms', 
                'Northing RMS': 'Y_rms',
                }
    df.rename(columns=namesdict, inplace=True)
    df.to_csv(path + "transects_%s.csv"%pth.split('_')[-1][:-4])


path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/saturation/'
paths = glob.glob(path + "DR_UTM_*.csv")

for pth in paths:
    df = pd.read_csv(pth)
    df = df[['Name', 'Easting', 'Northing', 'Averaging start', 'Easting RMS', 'Northing RMS']]
    A = [True if X in ['N', 'Ys', 'Yp', 'Yf'] else False for X in df['Name']]
    df = df[A]

    namesdict = {'Easting':'X',
                'Northing': 'Y',
                'Averaging start': 'BeginTime',
                'Easting RMS': 'X_rms', 
                'Northing RMS': 'Y_rms',
                }
    df.rename(columns=namesdict, inplace=True)
    df.to_csv(path + "transects_%s.csv"%pth.split('_')[-1][:-4])


#%% 
# Saturation on hillshade

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


#%% 
# TI vs saturation

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
    plt.savefig(f'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/DruidRun_sat_TI_{df.BeginTime[1][0:10]}.png')
    # plt.savefig(f'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/BaismanRun_sat_TI_{df.BeginTime[1][0:10]}.png')
    plt.show()

# %%
# multiplot of TI vs saturation

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


#%%

# assemble all saturation dataframes
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
    
    # add discharge to df 

    dfs.append(df)
dfall = pd.concat(dfs, axis=0)

#%%

path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/'
TIfile = "LSDTT/baltimore2015_DR1_TIfiltered.tif" # Druids Run
curvfile = "LSDTT/baltimore2015_DR1_CURV.bil" # Druids Run

# path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/'
# TIfile = "LSDTT/baltimore2015_BR_TIfiltered.tif" # Baisman Run
# curvfile = "LSDTT/10m_window/baltimore2015_BR_CURV.bil" # Baisman Run

# add filtered TI points
tis = rd.open(path+TIfile)
coords = [(x,y) for x, y in zip(dfall['X'], dfall['Y'])]
dfall['TI_filtered'] = [x for x in tis.sample(coords)]
tis.close()

# add sat val
sat_val_dict = {'N':0, 'Ys':1, 'Yp':2, 'Yf':3}
dfall['sat_val'] = dfall['Name'].apply(lambda x: sat_val_dict[x])

#%%
# add discharge
dfnew = dfall.merge(dfqug, on='date', how='left')
dfnew['Q'].fillna(0.0, inplace=True)

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


#%% cumulative TI-sat

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

# %%
# logistic regression to determine saturation

# X = df['TI_filtered'].values.reshape(-1, 1)

X = df[['TI_filtered', 'curv']]
y = df['sat_val'] > 1
y = y.astype(int)

clf = LogisticRegression(random_state=0).fit(X, y)

pred = clf.predict(X)
prob = clf.predict_proba(X)

sc = clf.score(X, y)



# %%

plt.figure()
plt.scatter(df['TI_filtered'], prob[:,0])
plt.scatter(df['TI_filtered'], prob[:,1])

# %%

fig, ax = plt.subplots()
sc = ax.scatter(df['TI_filtered'], 
                y + 0.05*np.random.randn(len(df)), 
                c='k',
                alpha=0.3,
                )
sc = ax.scatter(df['TI_filtered'], 
                pred + 0.05*np.random.randn(len(df)), 
                c=df['curv'], 
                cmap='coolwarm', 
                norm=colors.CenteredNorm())


# %%
df.to_csv(path+'sat_20220419.csv')
# %%
