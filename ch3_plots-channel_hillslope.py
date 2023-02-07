"""
Script to make figures for ch.3, first focusing on topographic analysis

"""

#%%

import numpy as np
import pandas as pd
import pickle
import glob
import rasterio as rd
from sklearn.neighbors import KernelDensity
from scipy.stats import binned_statistic
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from matplotlib.colors import LightSource

import cartopy as cp
import cartopy.crs as ccrs

from generate_colormap import get_continuous_cmap

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

def equalObs(x, nbin):
    """generate bins with equal number of observations
        source: https://www.statology.org/equal-frequency-binning-python/"""
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))

                     
#%%

path = 'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/'
name = 'denudation_piedmont_portenga_2019_fig_4.csv'
df_U = pd.read_csv(path+name) # (Mg km-2 yr-1)
df_U['U'] = df_U[' 10be_rate'] * 1e3 * 1e-6 * (1/2700)

#%% Channels and Hillslopes  on hillshade (projected coordinate method)



# path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/LSDTT/'
# name = 'baltimore2015_DR1_hs.bil' # hillshade
# name_ch = "baltimore2015_DR1_D_CN.csv"  # channels
# name_hds = "baltimore2015_DR1_Dsources.csv" # channel heads
# name_rge = "baltimore2015_DR1_RidgeData.csv" # ridges

path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/LSDTT/'
name = 'baltimore2015_BR_hs.bil'
name_ch = "baltimore2015_BR_D_CN.csv"
name_hds = "baltimore2015_BR_Dsources.csv"
name_rge = "baltimore2015_BR_RidgeData.csv"

src = rd.open(path + name) # hillshade
df = pd.read_csv(path + name_ch) # channels
df1 = pd.read_csv(path + name_hds) # channel heads
df2 = pd.read_csv(path + name_rge) # ridges

bounds = src.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
proj = src.crs
utm = 18

#%%

fig = plt.figure(figsize=(9,7))

# Set the projection to UTM zone 18
ax = fig.add_subplot(1, 1, 1, projection=ccrs.UTM(utm))

projected_coords = ax.projection.transform_points(ccrs.Geodetic(), df['longitude'], df['latitude'])
ax.scatter(projected_coords[:,0], projected_coords[:,1], s=0.5, c='b', transform=ccrs.UTM(utm)) #c=df['Stream Order'],

projected_coords = ax.projection.transform_points(ccrs.Geodetic(), df1['longitude'], df1['latitude'])
ax.scatter(projected_coords[:,0], projected_coords[:,1], s=3, c='r', transform=ccrs.UTM(utm)) #c=df['Stream Order'],

# inds = df2['basin_id'] == 99 # Druid Run
inds = df2['basin_id'] == 71 # Baisman Run
projected_coords = ax.projection.transform_points(ccrs.Geodetic(), df2['longitude'][inds], df2['latitude'][inds])
ax.scatter(projected_coords[:,0], projected_coords[:,1], s=0.5, c='gold', transform=ccrs.UTM(utm)) #c='g',

# projected_coords = ax.projection.transform_points(ccrs.Geodetic(), df2['longitude'], df2['latitude'])
# ax.scatter(projected_coords[:,0], projected_coords[:,1], s=1, c=df2['basin_id'], transform=ccrs.UTM(utm)) #c='g',


# Limit the extent of the map to a small longitude/latitude range.
ax.set_extent(Extent, crs=ccrs.UTM(utm))
ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
cs = ax.imshow(src.read(1), cmap='binary', vmin=100, #cmap='plasma', vmin=-0.1, vmax=0.1, #
               extent=Extent, transform=ccrs.UTM(utm), origin="upper")
# plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/DruidRun_channels.png')
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/OregonRidge_channels.png')
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
# violin plots of hillslope length and relief

pos   = [1, 2]
label = ['Druids Run', 'Baisman Run']

Lh = [df_ht_DR['Lh'], df_ht_BR['Lh']]
R = [df_ht_DR['R'], df_ht_BR['R']]
fig, axs = plt.subplots(ncols=2, figsize=(8,5))
axs[0].violinplot(Lh, pos, vert=True, showmeans=True)
axs[0].set_ylim((-10,1200))
axs[0].set_xticks(pos)
axs[0].set_xticklabels(label)
axs[0].set_ylabel('Length [m]')
axs[0].set_title('Hillslope Length')

axs[1].violinplot(R, pos, vert=True, showmeans=True)
# axs[1].set_ylim((-10,1200))
axs[1].set_xticks(pos)
axs[1].set_xticklabels(label)
axs[1].set_title('Hillslope Relief')
axs[1].set_ylabel('Height [m]')
plt.show()
fig.tight_layout()
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/Lh_R_violinplot.png')


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

#%% 
# Violin plots of ridgetop curvature

cBR = df_ht_BR['Cht'] > -1
cDR = df_ht_DR['Cht'] > -1

pos   = [1, 2]
label = ['Druids Run', 'Baisman Run']

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

# %% 
# calculate some statistics

cBR = df_ht_BR['Cht'] > -1
cDR = df_ht_DR['Cht'] > -1

DR = []
BR = []
Cht_stats = {}
BR.append(np.median(df_ht_BR['Cht'][cBR]).round(5))
DR.append(np.median(df_ht_DR['Cht'][cDR]).round(5))
# Cht_stats['Mean_BR'] = np.mean(df_ht_BR['Cht'][cBR]).round(5)
# Cht_stats['Mean_DR'] = np.mean(df_ht_DR['Cht'][cDR]).round(5)

# log-transformed 
mean_log = np.mean(np.log(-df_ht_DR['Cht'][cDR]))
std_log = np.std(np.log(-df_ht_DR['Cht'][cDR]))
DR.append(-np.exp(mean_log - std_log))
DR.append(-np.exp(mean_log + std_log))
DR.append(-np.exp(mean_log))

mean_log = np.mean(np.log(-df_ht_BR['Cht'][cBR]))
std_log = np.std(np.log(-df_ht_BR['Cht'][cBR]))
BR.append(-np.exp(mean_log - std_log))
BR.append(-np.exp(mean_log + std_log))
BR.append(-np.exp(mean_log))

# data = np.array(list(zip(DR, BR)))
df_HT_stats = pd.DataFrame([DR, BR], index=['DR', 'BR'], columns=['Med','log_high', 'log_low', 'log_mean'])
df_HT_stats.to_csv(path+'df_HT_stats.csv')

E = df_U['U'][2]
rho_ratio = 2

df_D_stats = (rho_ratio*E)/(-df_HT_stats)


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


# %% plot basin

dx = bsn.transform[0]
bounds = bsn.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]

plt.figure()
plt.imshow(basin,
            origin="upper", 
            extent=Extent,
            cmap='viridis',
            )
plt.show()


# %% Chi analysis visualization

conc = '0.6'

path1 = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/LSDTT/'
path2 = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/LSDTT/'

# name_chi_DR = "baltimore2015_DR1_chi_data_map.csv"
# name_chi_BR = "baltimore2015_BR_chi_data_map.csv"

name_chi_DR = "baltimore2015_DR1_%s_MChiSegmented.csv"%conc
name_chi_BR = "baltimore2015_BR_%s_MChiSegmented.csv"%conc

df_chi_DR = pd.read_csv(path1 + name_chi_DR)
df_chi_BR = pd.read_csv(path2 + name_chi_BR)

# %% Chi-elevation plots

fig, axs = plt.subplots()
sc = axs.scatter(df_chi_DR['chi'], df_chi_DR['elevation'], c=df_chi_DR['m_chi'], s=3)
axs.set_xlabel(r'$\chi$ (m)')
axs.set_ylabel('Elevation (m)')
axs.set_title('Druids Run: m/n=%s'%conc)
fig.colorbar(sc, label=r'log10 $k_{sn}$')
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/chi_DR_%s.png'%conc)

fig, axs = plt.subplots()
sc = axs.scatter(df_chi_BR['chi'], df_chi_BR['elevation'], c=df_chi_BR['m_chi'], s=3)
axs.set_xlabel(r'$\chi$ (m)')
axs.set_ylabel('Elevation')
axs.set_title('Baisman Run: m/n=%s'%conc)
fig.colorbar(sc, label=r'log10 $k_{sn}$')
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/chi_BR_%s.png'%conc)

#%% flow distance-elevation plots 

fig, axs = plt.subplots()
sc = axs.scatter(df_chi_DR['flow_distance'], df_chi_DR['elevation'], c=df_chi_DR['m_chi'], s=3)
axs.set_xlabel(r'flow distance (m)')
axs.set_ylabel('Elevation (m)')
axs.set_title('Druids Run')
fig.colorbar(sc, label=r'log10 $k_{sn}$')
# plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/flow_length_DR.png')

fig, axs = plt.subplots()
sc = axs.scatter(df_chi_BR['flow_distance'], df_chi_BR['elevation'], c=df_chi_BR['m_chi'], s=3)
axs.set_xlabel(r'flow distance (m)')
axs.set_ylabel('Elevation')
axs.set_title('Baisman Run')
fig.colorbar(sc, label=r'log10 $k_{sn}$')
# plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/flow_length_BR.png')

# %%

plt.figure()
df_chi_BR['m_chi'].plot.density(color='r', label='Baisman')
df_chi_DR['m_chi'].plot.density(color='b', label='Druids')
plt.xlabel('log10 $k_{sn}$')
plt.legend(frameon=False)
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/ksn_segment_distr.png')


#%%
plt.figure()
df_chi_BR['drainage_area'].plot.cdf(color='r', label='Baisman')
df_chi_DR['drainage_area'].plot.density(color='b', label='Druids')
# plt.xlabel('log10 $k_{sn}$')
plt.legend(frameon=False)


Q75_BR = np.quantile(df_chi_BR['drainage_area'], 0.75)
Q75_DR = np.quantile(df_chi_DR['drainage_area'], 0.75)


# %% map Ksn


# path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/LSDTT/'
# name = 'baltimore2015_DR1_hs.bil' # hillshade
# df = df_chi_DR

path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/LSDTT/'
name = 'baltimore2015_BR_hs.bil'
df = df_chi_BR

src = rd.open(path + name) # hillshade

bounds = src.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
proj = src.crs
utm = 18

fig = plt.figure(figsize=(9,7))

# Set the projection to UTM zone 18
ax = fig.add_subplot(1, 1, 1, projection=ccrs.UTM(utm))

projected_coords = ax.projection.transform_points(ccrs.Geodetic(), df['longitude'], df['latitude'])
sc = ax.scatter(projected_coords[:,0], 
            projected_coords[:,1], 
            s=2, 
            c=df['m_chi'], 
            transform=ccrs.UTM(utm)
            )

# Limit the extent of the map to a small longitude/latitude range.
ax.set_extent(Extent, crs=ccrs.UTM(utm))
# ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
ax.imshow(src.read(1), cmap='binary', vmin=100, #cmap='plasma', vmin=-0.1, vmax=0.1, #
               extent=Extent, transform=ccrs.UTM(utm), origin="upper")
fig.colorbar(sc, label='log10 $k_{sn}$')
# plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/DruidRun_ksn.png')
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/OregonRidge_ksn.png')
plt.show()



fig = plt.figure(figsize=(9,7))

# Set the projection to UTM zone 18
ax = fig.add_subplot(1, 1, 1, projection=ccrs.UTM(utm))

projected_coords = ax.projection.transform_points(ccrs.Geodetic(), df['longitude'], df['latitude'])
sc = ax.scatter(projected_coords[:,0], 
            projected_coords[:,1], 
            s=2, 
            c=df['chi'], 
            cmap='plasma',
            transform=ccrs.UTM(utm)
            )
ax.set_extent(Extent, crs=ccrs.UTM(utm))
# ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
ax.imshow(src.read(1), 
            cmap='binary', 
            vmin=100,
            extent=Extent, 
            transform=ccrs.UTM(utm), 
            origin="upper")
fig.colorbar(sc, label='$\chi$')
# plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/DruidRun_chi.png')
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/OregonRidge_chi.png')
plt.show()
# %%
