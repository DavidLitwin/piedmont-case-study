

import numpy as np
import pandas as pd
import pickle
import rasterio as rd
from random import sample 
from sklearn.neighbors import KernelDensity
from scipy.stats import binned_statistic
import statsmodels.api as sm

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

import cartopy as cp
import cartopy.crs as ccrs

from landlab import imshow_grid, RasterModelGrid
from landlab.components import LakeMapperBarnes, FlowAccumulator, HeightAboveDrainageCalculator, DrainageDensity
from landlab.utils import get_watershed_mask

from generate_colormap import get_continuous_cmap

def equalObs(x, nbin):
    """generate bins with equal number of observations
        source: https://www.statology.org/equal-frequency-binning-python/"""
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))


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

path = 'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/'
name = 'denudation_piedmont_portenga_2019_fig_4.csv'
df_U = pd.read_csv(path+name) # (Mg km-2 yr-1)
df_U['U'] = df_U[' 10be_rate'] * 1e3 * 1e-6 * (1/2700)

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

#%% Calculate kernel density of hillslope length and relief

path1 = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/LSDTT/'
path2 = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/LSDTT/'

name_rge_DR = "baltimore2015_DR1_RidgeData.csv"
name_rge_BR = "baltimore2015_BR_RidgeData.csv"

name_ht_DR = "baltimore2015_DR1_HilltopData_TN.csv"
name_ht_BR = "baltimore2015_BR_HilltopData_TN.csv"

df_ht_DR = pd.read_csv(path1 + name_ht_DR)
df_ht_DR = df_ht_DR[df_ht_DR['BasinID']==99]
df_ht_BR = pd.read_csv(path2 + name_ht_BR)
df_ht_BR = df_ht_BR[df_ht_BR['BasinID']==71]
#%%

# calculate kernel density
xx_DR, yy_DR, zz_DR = kde2D(df_ht_DR['Lh']/max(df_ht_DR['Lh']), df_ht_DR['R']/max(df_ht_DR['R']), 0.05, xbins=200j, ybins=200j)
xx_BR, yy_BR, zz_BR = kde2D(df_ht_BR['Lh']/max(df_ht_BR['Lh']), df_ht_BR['R']/max(df_ht_BR['R']), 0.05, xbins=200j, ybins=200j)

#%% plot kernel density

# hex_colors = ['000000', '083D77', 'F95738', 'EE964B', 'F4D35E', 'EBEBD3'] # based on https://coolors.co/palette/083d77-ebebd3-f4d35e-ee964b-f95738
# hex_decs = [0, 0.1, 0.6, 0.8, 0.9, 1.0 ]
hex_colors = ['FFFFFF', '00B4D8', '03045E']
hex_decs = [0, 0.5, 1.0]

cmap1 = get_continuous_cmap(hex_colors, float_list=hex_decs)

# plot kernel density
fig, axs = plt.subplots(ncols=2, figsize=(8,4))
axs[0].pcolormesh(xx_DR*max(df_ht_DR['Lh']), yy_DR* max(df_ht_DR['R']), zz_DR, cmap=cmap1, shading='auto')
axs[0].set_xlabel('Distance to channel (m)')
axs[0].set_ylabel('Height above channel (m)')
axs[0].set_title('Druids Run')
axs[0].set_xlim((0,1000))
axs[0].set_ylim((0,25))
# axs[0].colorbar(label='Density')
pc = axs[1].pcolormesh(xx_BR*max(df_ht_BR['Lh']), yy_BR*max(df_ht_BR['R']), zz_BR, cmap=cmap1, shading='auto')
axs[1].set_xlabel('Distance to channel (m)')
axs[1].set_ylabel('Height above channel (m)')
axs[1].set_title('Baisman Run')
axs[1].set_xlim((0,1000))
axs[1].set_ylim((0,25))
# fig.colorbar(pc, label='Density')
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/hillslopelen_relief.png')

plt.show()
# %% 
# boxplots of hillslope length and relief

Lh = [df_ht_DR['Lh'], df_ht_BR['Lh']]
R = [df_ht_DR['R'], df_ht_BR['R']]
fig, axs = plt.subplots(ncols=2, figsize=(8,5))
axs[0].boxplot(Lh, whis=(5,95), vert=True, labels=['Druids Run', 'Baisman Run'])
axs[0].set_ylim((-10,1200))
axs[0].set_ylabel('Length [m]')
axs[0].set_title('Hillslope Length')
axs[1].boxplot(R, whis=(5,95), vert=True, labels=['Druids Run', 'Baisman Run'])
axs[1].set_title('Hillslope Relief')
axs[1].set_ylabel('Height [m]')
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/Lh_R_boxplot.png')


#%% 
# boxplots of ridgetop curvature

plt.figure(figsize=(4,4))
plt.boxplot([df_ht_DR['Cht'], df_ht_BR['Cht']], 
            whis=(5,95), 
            vert=True, 
            widths=0.8,
            labels=['Druids Run', 'Baisman Run'])
plt.ylim(( -0.1, 0.005))
plt.ylabel('Ridgetop Curvature [1/m]')
plt.tight_layout()
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/Cht_boxplot.png')


# %% 
# Violin plots of effective curvature R/Lh^2

C_DR = df_ht_DR['R']/df_ht_DR['Lh']**2
C_BR = df_ht_BR['R']/df_ht_BR['Lh']**2

cBR = C_BR < 0.1
cDR = C_DR < 0.1

DR = []
BR = []
BR.append(-np.median(C_BR[cBR]).round(5))
DR.append(-np.median(C_DR[cDR]).round(5))

# log-transformed 
mean_log = np.mean(np.log(C_DR[cDR]))
std_log = np.std(np.log(C_DR[cDR]))
DR.append(-np.exp(mean_log - std_log))
DR.append(-np.exp(mean_log + std_log))
DR.append(-np.exp(mean_log))

mean_log = np.mean(np.log(C_BR[cBR]))
std_log = np.std(np.log(C_BR[cBR]))
BR.append(-np.exp(mean_log - std_log))
BR.append(-np.exp(mean_log + std_log))
BR.append(-np.exp(mean_log))

df_Ceff_stats = pd.DataFrame([DR, BR], index=['DR', 'BR'], columns=['Med','log_high', 'log_low', 'log_mean'])

#%%


pos   = [1, 2]
label = ['Druids Run', 'Baisman Run']

fig, ax = plt.subplots(figsize=(4,4))
ax.violinplot([np.log10(C_DR), np.log10(C_BR)],
            pos,
            vert=True, 
            showmeans=True,
            )
# plt.ylim(( -0.1, 0.005))
ax.set_ylabel('Log10 Curvature R/Lh2')
ax.set_xticks(pos)
ax.set_xticklabels(label)
fig.tight_layout()
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/HillCurv_violinplot.png')
plt.show()

# %% 
# Violin plots of ridgetop and effective curvature together
Ceff_BR = df_ht_BR['R']/df_ht_BR['Lh']**2
Ceff_DR = df_ht_DR['R']/df_ht_DR['Lh']**2

cht_BR = df_ht_BR['Cht'] > -1
cht_DR = df_ht_DR['Cht'] > -1
# cBR = Ceff_BR < 0.1
# cDR = Ceff_DR < 0.1

pos   = [1, 2]
label = ['Druids Run', 'Baisman Run']

fig, axs = plt.subplots(figsize=(8,4), ncols=2)
axs[0].violinplot([np.log10(-df_ht_DR['Cht'][cht_DR]), np.log10(-df_ht_BR['Cht'][cht_BR])],
            pos,
            vert=True, 
            showmeans=True,
            )
# axs[0].set_yscale('log')
axs[0].set_ylim((-7,1))
axs[0].set_ylabel(r'$\log_{10}(-C_{ht})$')
axs[0].set_xticks(pos)
axs[0].set_xticklabels(label)
axs[0].set_title('Hilltop Curvature')

axs[1].violinplot([np.log10(Ceff_DR), np.log10(Ceff_BR)],
            pos,
            vert=True, 
            showmeans=True,
            )
# axs[1].set_yscale('log')
axs[1].set_ylim((-7,1))
axs[1].set_ylabel(r'$\log_{10}(R_h/L_h^2)$')
axs[1].set_xticks(pos)
axs[1].set_xticklabels(label)
axs[1].set_title('Effective Hillslope Curvature')
fig.tight_layout()
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/HT_Eff_Curv_violinplot.png')



cBR = df_ht_BR['Cht'] > -1
cDR = df_ht_DR['Cht'] > -1

fig, ax = plt.subplots(figsize=(4,4))
ax.violinplot([df_ht_DR['Cht'][cDR], df_ht_BR['Cht'][cBR]],
            pos,
            vert=True, 
            showmeans=True,
            )
# plt.ylim(( -0.1, 0.005))
ax.set_ylabel('Ridgetop Curvature [1/m]')
ax.set_xticks(pos)
ax.set_xticklabels(label)
fig.tight_layout()
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/Cht_violinplot.png')


# %% Violin plots of E*

cBR = df_ht_BR['E_Star'] < 1e6
cDR = df_ht_DR['E_Star'] < 1e6

pos   = [1, 2]
label = ['Druids Run', 'Baisman Run']

fig, ax = plt.subplots(figsize=(4,4))
ax.violinplot([np.log10(df_ht_DR['E_Star'][cDR]), np.log10(df_ht_BR['E_Star'][cBR])],
            pos,
            vert=True, 
            showmeans=True,
            )
ax.axhline(y=1, linestyle='--', color='k', label='E*=1')
# plt.ylim(( -0.1, 0.005))
ax.set_ylabel(r'$log_{10} (E^*)$')
ax.set_xticks(pos)
ax.set_xticklabels(label)
ax.legend(loc='lower center', )
fig.tight_layout()
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/Estar_violinplot.png')




#%% slope-area of channels

# Druids Run
path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/LSDTT/'
somask_name = 'baltimore2015_DR1_D_SO.bil' # stream order raster
slope_name = 'baltimore2015_DR1_SLOPE.bil'
area_name = 'baltimore2015_DR1_d8_area.bil'
basin_name = 'baltimore2015_DR1_AllBasins.bil'

somsk = rd.open(path + somask_name)
ar = rd.open(path + area_name)
slp = rd.open(path + slope_name)
bsn = rd.open(path + basin_name)

streams = somsk.read(1) > 0
basin = bsn.read(1) > 0 
mask = np.logical_and(streams, basin)
area_DR = ar.read(1)[mask]
slope_DR = slp.read(1)[mask]

area_DR = area_DR[slope_DR>0]
slope_DR = slope_DR[slope_DR>0]


# Baisman Run
path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/LSDTT/'
somask_name = 'baltimore2015_BR_D_SO.bil'
slope_name = 'baltimore2015_BR_SLOPE.bil' 
area_name = 'baltimore2015_BR_d8_area.bil'
basin_name = 'baltimore2015_BR_AllBasins.bil'

somsk = rd.open(path + somask_name)
ar = rd.open(path + area_name)
slp = rd.open(path + slope_name)
bsn = rd.open(path + basin_name)

streams = somsk.read(1) > 0
basin = bsn.read(1) > 0 
mask = np.logical_and(streams, basin)
area_BR = ar.read(1)[mask]
slope_BR = slp.read(1)[mask]

area_BR = area_BR[slope_BR>0]
slope_BR = slope_BR[slope_BR>0]


#%% slope-area relationship

slope = slope_BR
area = area_BR

# linear model
X = np.log(area).reshape((-1,1))
y = np.log(slope)
# X = np.log(bin_center).reshape((-1,1))
# y = np.log(s_mean)
x = sm.add_constant(X)
model = sm.OLS(y, x).fit()
df_stats = (model.summary2().tables[1])

df_stats = df_stats.T
df_stats.drop(['Std.Err.', 't', 'P>|t|'], inplace=True)
df_stats['concavity'] = -df_stats['x1']
df_stats['steepness'] = np.exp(df_stats['const'])
# df_stats['m_guess'] = 0.5
# df_stats['n'] = df_stats['m_guess']/df_stats['concavity']
df_stats['n_guess'] = 1
df_stats['K'] = df_U['U'][2]/df_stats['steepness']**(df_stats['n_guess'])

#%% plot slope-area and fitted line

# get binned data
areax = equalObs(area, 30)
s_mean, bin_edge, binnum = binned_statistic(area, slope, statistic='mean', bins=areax)
s_std, _, _ = binned_statistic(area, slope, statistic='std', bins=areax)
bin_center = bin_edge[:-1] + np.diff(bin_edge)

#get fitted line
X_pred = np.geomspace(np.min(areax),np.max(areax), 500) #.reshape(-1,1)
y_pred = model.predict()

fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(area, slope, alpha=0.1, s=1.5)
ax.errorbar(bin_center, s_mean, s_std, linestyle='None', marker='o', color='k', markersize=4)
ax.plot(np.exp(X), np.exp(y_pred), 'r--')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim((1e-4,0.3))
ax.set_ylabel('Slope (m/m)')
ax.set_xlabel('Area (m)')
fig.tight_layout()
# plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/slope_area_DR.png')
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/slope_area_BR.png')


#%%

x = equalObs(area_DR, 30)
s_mean_DR, bin_edge_DR, binnum_DR = binned_statistic(area_DR, slope_DR, statistic='mean', bins=x)
s_std_DR, _, _ = binned_statistic(area_DR, slope_DR, statistic='std', bins=x)
bin_center_DR = bin_edge_DR[:-1] + np.diff(bin_edge_DR)

x = equalObs(area_BR, 30)
s_mean_BR, bin_edge_BR, binnum_BR = binned_statistic(area_BR, slope_BR, statistic='mean', bins=x)
s_std_BR, _, _ = binned_statistic(area_BR, slope_BR, statistic='std', bins=x)
bin_center_BR = bin_edge_BR[:-1] + np.diff(bin_edge_BR)

fig, ax = plt.subplots(figsize=(6,5))
ax.errorbar(bin_center_DR, s_mean_DR, s_std_DR, linestyle='None', marker='o', color='r', markersize=4, label='Druid')
ax.errorbar(bin_center_BR, s_mean_BR, s_std_BR, linestyle='None', marker='o', color='b', markersize=4, label='Baisman')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Slope (m/m)')
ax.set_xlabel('Area (m)')
fig.legend(frameon=False)
fig.tight_layout()