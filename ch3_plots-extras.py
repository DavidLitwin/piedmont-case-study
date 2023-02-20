

import numpy as np
import pandas as pd
import pickle
import rasterio as rd
from random import sample 
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

import cartopy as cp
import cartopy.crs as ccrs

from landlab import imshow_grid, RasterModelGrid
from landlab.components import LakeMapperBarnes, FlowAccumulator, HeightAboveDrainageCalculator, DrainageDensity
from landlab.utils import get_watershed_mask


def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
    """Build 2D kernel density estimate (KDE).
        source: https://stackoverflow.com/a/41639690
    """

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)

#%%


path = "C:/Users/dgbli/Documents/ArcGIS/Projects/Baisman_Run/"
nodata = -9999

# topography
topofile = "baltimore2015_BR_ws.tif"
zf = rd.open(path+topofile)
z = zf.read(1).astype(float)
z = np.ma.masked_array(z, mask=z==0) # for some reason nodata is 0 here
# zf.close()
# transform: zf.transform

# fitted (10 m) slope
slopefile = "baltimore2015_BR_SLOPE_ws.tif"
sf = rd.open(path+slopefile)
slope = sf.read(1).astype(float)
slope = np.ma.masked_array(slope, mask=slope==nodata)
# sf.close()

# d-infinity area
areafile = "baltimore2015_BR_dinf_ws.tif"
af = rd.open(path+areafile)
area = af.read(1).astype(float)
area = np.ma.masked_array(area, mask=area==nodata)
# af.close()

#%% Hillshade

plt.figure()
ls = LightSource(azdeg=135, altdeg=45)
plt.imshow(
            ls.hillshade(z, 
                vert_exag=2, 
                dx=zf.transform[0], 
                dy=zf.transform[0]), 
            origin="upper", 
            extent=(zf.bounds[0], zf.bounds[2], zf.bounds[3], zf.bounds[1]),
            cmap='gray',
            )
plt.show()

# # to get full coordinates of points:
# Xg, Yg = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
# X, Y = zf.transform * (Xg, Yg)

# %% Topographic index

TI = np.log(area/(slope * zf.transform[0]))

plt.figure()
plt.imshow(TI,
            origin="upper", 
            extent=(zf.bounds[0], zf.bounds[2], zf.bounds[3], zf.bounds[1]),
            cmap='viridis',
            )
plt.show()


#%%

mg = pickle.load(open(path+'modelgrid.p', 'rb'))

# %%

hand = mg.at_node['height_above_drainage__elevation']
mask = mg.at_node['watershed_mask']

plt.figure()
handm = np.ma.masked_array(hand, ~mask)
imshow_grid(mg, handm, color_for_closed='k')
plt.show()

plt.figure()
hill_len = mg.at_node['surface_to_channel__minimum_distance']
hill_lenm = np.ma.masked_array(hill_len, ~mask)
imshow_grid(mg, hill_lenm, color_for_closed='k')
plt.show()

# %%

plt.figure()
plt.scatter(hill_lenm, handm, alpha=0.005, s=2)

plt.figure()
plt.hist(handm, bins=100)

# Failed attempt at sampling just the areas with no flow upslope
# area = mg.at_node['drainage_area']
# a0 = np.sort(np.unique(area))[1]
# sources = np.logical_and(area>0, area<2*a0)

# plt.figure()
# plt.scatter(hill_lenm[sources], handm[sources], alpha=0.05, s=4)

# plt.figure()
# imshow_grid(mg, sources, color_for_closed='k')
# plt.show()

#%% Create kernel density plot from subsample of data

subint = sample(range(len(handm)), 100000)
xx, yy, zz = kde2D(hill_lenm[subint], handm[subint], 3.0)

# plot kernel density
plt.figure()
plt.pcolormesh(xx, yy, zz)
plt.xlabel('Distance to channel (m)')
plt.ylabel('Height above channel (m)')
plt.colorbar(label='Density')
plt.show()

# plot kernel density, truncated to threshold (eventually truncate to 95%, or 99%?)
zzm = np.ma.masked_array(zz, zz<1e-5)
plt.figure()
plt.pcolormesh(xx, yy, zzm)
plt.xlabel('Distance to channel (m)')
plt.ylabel('Height above channel (m)')
plt.colorbar(label='Density')
plt.show()

#%% alternate baseflow separation and event identification

# path_DR = "C:/Users/dgbli/Documents/Research/Soldiers Delight/data_processed/Gianni_event_DR/"
# path_BR = "C:/Users/dgbli/Documents/Research/Oregon Ridge/data_processed/Gianni_event_BAIS/"

path_DR = "C:/Users/dgbli/Documents/Research/Soldiers Delight/data_processed/Manual_event_DR/"
path_BR = "C:/Users/dgbli/Documents/Research/Oregon Ridge/data_processed/Manual_event_BAIS/"


file_DR = 'Druids Run.csv'
file_BR = 'Baisman Run.csv'
df_DR = pd.read_csv(path_DR+file_DR)
df_BR = pd.read_csv(path_BR+file_BR)

df_DR['date'] = pd.to_datetime(df_DR[['year', 'month', 'day', 'hour', 'minute', 'second']])
df_BR['date'] = pd.to_datetime(df_BR[['year', 'month', 'day', 'hour', 'minute', 'second']])
df_DR.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)
df_BR.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)


# event_names = ['Q_start', 'Q_end', 'P_start', 'P_end']
file = 'StartFinishFlowStartFinishRain.csv'
df_DR_event = pd.read_csv(path_DR+file) #, names=event_names
df_BR_event = pd.read_csv(path_BR+file)

# %%

fig, ax = plt.subplots()
ax.plot(df_DR.date, df_DR['flow (mm/day)'], 'k-')
ax.plot(df_DR.date, df_DR['baseflow (mm/day)'], 'b-')
ax.scatter(df_DR['date'].iloc[df_DR_event['Q_start']], df_DR['flow (mm/day)'].iloc[df_DR_event['Q_start'].values], c='g')
ax.scatter(df_DR['date'].iloc[df_DR_event['Q_end']], df_DR['flow (mm/day)'].iloc[df_DR_event['Q_end'].values], c='r')
ax.set_yscale('log')
ax.set_ylabel('Q (mm/day)')
ax1 = ax.twinx()
ax1.plot(df_DR.date, df_DR['rain (mm/day)'])
ax1.set_ylim(2*np.max(df_DR['rain (mm/day)']), 0)
ax1.set_ylabel('P (mm/day)')
fig.autofmt_xdate()
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/DR_Q_P.png')
plt.show()

#%%
fig, ax = plt.subplots()
ax.plot(df_BR.date, df_BR['flow (mm/day)'], 'k-')
ax.plot(df_BR.date, df_BR['baseflow (mm/day)'], 'b-')
ax.scatter(df_BR['date'].iloc[df_BR_event['Q_start']], df_BR['flow (mm/day)'].iloc[df_BR_event['Q_start'].values], c='g')
ax.scatter(df_BR['date'].iloc[df_BR_event['Q_end']], df_BR['flow (mm/day)'].iloc[df_BR_event['Q_end'].values], c='r')
ax.set_yscale('log')
ax.set_ylabel('Q (mm/day)')
ax1 = ax.twinx()
ax1.plot(df_BR.date, df_BR['rain (mm/day)'])
ax1.set_ylim(2*np.max(df_BR['rain (mm/day)']), 0)
ax1.set_ylabel('P (mm/day)')
fig.autofmt_xdate()
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/BR_Q_P.png')
plt.show()


# %%

df_BR['quickflow (mm)'] = (df_BR['flow (mm/day)'] - df_BR['baseflow (mm/day)'])/(24*4) # convert from mm/day to mm per time interval (15 min)
df_DR['quickflow (mm)'] = (df_DR['flow (mm/day)'] - df_DR['baseflow (mm/day)'])/(24*4)
df_BR['precip (mm)'] = df_BR['rain (mm/day)']/(24*4)
df_DR['precip (mm)'] = df_DR['rain (mm/day)']/(24*4)

df_BR_event['P'] = np.nan
df_BR_event['Qf'] = np.nan
df_BR_event['start_date'] = np.nan
for i in df_BR_event.index:
    df_BR_event['P'].iloc[i] = df_BR['precip (mm)'].iloc[df_BR_event['P_start'][i]:df_BR_event['P_end'][i]].sum()
    df_BR_event['Qf'].iloc[i] = df_BR['quickflow (mm)'].iloc[df_BR_event['Q_start'][i]:df_BR_event['Q_end'][i]].sum()
    df_BR_event['start_date'].iloc[i] = df_BR['date'].iloc[df_BR_event['P_start']]


df_DR_event['P'] = np.nan
df_DR_event['Qf'] = np.nan
df_DR_event['start_date'] = np.nan
for i in df_DR_event.index:
    df_DR_event['P'].iloc[i] = df_DR['precip (mm)'].iloc[df_DR_event['P_start'][i]:df_DR_event['P_end'][i]].sum()
    df_DR_event['Qf'].iloc[i] = df_DR['quickflow (mm)'].iloc[df_DR_event['Q_start'][i]:df_DR_event['Q_end'][i]].sum()
    df_DR_event['start_date'].iloc[i] = df_DR['date'].iloc[df_DR_event['P_start']]


#%%

# all
fig, axs = plt.subplots(ncols=2, figsize=(8,5))
axs[0].scatter(df_BR_event['P'], df_BR_event['Qf'], alpha=0.5, c='b')
axs[0].axline([0,0], [1,1], color='k', linestyle='--')
axs[0].set_ylim((-1,50))
axs[0].set_xlim((-3,110))
axs[0].set_ylabel('Event Q (mm)')
axs[0].set_xlabel('Event P (mm)')
axs[0].set_title('Baisman Run')

axs[1].scatter(df_DR_event['P'], df_DR_event['Qf'], alpha=0.5, c='r')
axs[1].axline([0,0], [1,1], color='k', linestyle='--')
axs[1].set_ylim((-1,50))
axs[1].set_xlim((-3,110))
axs[1].set_ylabel('Event Q (mm)')
axs[1].set_xlabel('Event P (mm)')
axs[1].set_title('Druids Run')
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/Event_RR.png')
plt.show()