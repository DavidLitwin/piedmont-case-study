"""
Script to make figures for ch.3, first focusing on topographic analysis

"""

#%%

import numpy as np
import pandas as pd
import rasterio as rd
from sklearn.neighbors import KernelDensity
from scipy.stats import norm, ranksums, binned_statistic
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

# figpath = 'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/'
# path = 'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/'
# path_DR = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/LSDTT/'
# path_BR = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/LSDTT/'

figpath = '/Users/dlitwin/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/'
path = '/Users/dlitwin/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/'
path_DR = '/Users/dlitwin/Documents/Research/Soldiers Delight/data/LSDTT/'
path_BR = '/Users/dlitwin/Documents/Research/Oregon Ridge/data/LSDTT/'

#%%


name = 'denudation_piedmont_portenga_2019_fig_4.csv'
df_U = pd.read_csv(path+name) # (Mg km-2 yr-1)
df_U['U'] = df_U[' 10be_rate'] * 1e3 * 1e-6 * (1/2700)

#%% Channels and Hillslopes  on hillshade (projected coordinate method)

name = 'baltimore2015_DR1_hs.bil' # hillshade
name_ch = "baltimore2015_DR1_D_CN.csv"  # channels
name_hds = "baltimore2015_DR1_Dsources.csv" # channel heads
name_rge = "baltimore2015_DR1_RidgeData.csv" # ridges

# path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/LSDTT/'
# name = 'baltimore2015_BR_hs.bil'
# name_ch = "baltimore2015_BR_D_CN.csv"
# name_hds = "baltimore2015_BR_Dsources.csv"
# name_rge = "baltimore2015_BR_RidgeData.csv"

src = rd.open(path_DR + name) # hillshade
df = pd.read_csv(path_DR + name_ch) # channels
df1 = pd.read_csv(path_DR + name_hds) # channel heads
df2 = pd.read_csv(path_DR + name_rge) # ridges

bounds = src.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
proj = src.crs
utm = 18

#%% get average elevations

name_DR = 'baltimore2015_DR1.bil'
basin_DR = 'baltimore2015_DR1_AllBasins.bil'

name_BR = 'baltimore2015_BR.bil'
basin_BR = 'baltimore2015_BR_AllBasins.bil'

src_DR = rd.open(path_DR + name_DR)
src_BR = rd.open(path_BR + name_BR)
elev_DR = src_DR.read(1)
elev_BR = src_BR.read(1)

bsn_BR = rd.open(path_BR + basin_BR)
bsn_DR = rd.open(path_DR + basin_DR)
basin_DR = bsn_DR.read(1) > 0 
basin_BR = bsn_BR.read(1) > 0 


elev_mean_DR = np.mean(elev_DR[basin_DR])
elev_mean_BR = np.mean(elev_BR[basin_BR])

plt.figure()
plt.imshow(basin_DR,
            origin="upper", 
            extent=Extent,
            cmap='viridis',
            )
plt.show()



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

# Limit the extent of the map to a small longitude/latitude range.
ax.set_extent(Extent, crs=ccrs.UTM(utm))
ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
cs = ax.imshow(src.read(1), cmap='binary', vmin=100, #cmap='plasma', vmin=-0.1, vmax=0.1, #
               extent=Extent, transform=ccrs.UTM(utm), origin="upper")
# plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/DruidRun_channels.png')
# plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/OregonRidge_channels.png')
plt.show()

#%% Calculate kernel density of hillslope length and relief


name_rge_DR = "baltimore2015_DR1_RidgeData.csv"
name_rge_BR = "baltimore2015_BR_RidgeData.csv"

name_ht_DR = "baltimore2015_DR1_HilltopData_TN.csv"
name_ht_BR = "baltimore2015_BR_HilltopData_TN.csv"

df_ht_DR = pd.read_csv(path_DR + name_ht_DR)
df_ht_DR = df_ht_DR[df_ht_DR['BasinID']==99]
df_ht_BR = pd.read_csv(path_BR + name_ht_BR)
df_ht_BR = df_ht_BR[df_ht_BR['BasinID']==71]

#%% violin plots of hillslope length and relief

pos   = [1, 2]
label = ['Druids Run', 'Baisman Run']
clrs = ['firebrick', 'royalblue']

Lh = [df_ht_DR['Lh'].values, df_ht_BR['Lh'].values]
R = [df_ht_DR['R'].values, df_ht_BR['R'].values]

Lh_stat = ranksums(Lh[0], Lh[1])
R_stat = ranksums(R[0], R[1])

#%%
fig, axs = plt.subplots(ncols=2, figsize=(8,5))
parts = axs[0].violinplot(Lh, pos, vert=True, showmeans=False, showmedians=True,
        showextrema=True)
for pc, color in zip(parts['bodies'], clrs):
    pc.set_facecolor(color)
    pc.set_alpha(0.8)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp = parts[partname]
    vp.set_edgecolor("k")
    vp.set_linewidth(1)
DRq1, DRmed, DRq3 = np.percentile(df_ht_DR['Lh'].values, [25, 50, 75])
BRq1, BRmed, BRq3 = np.percentile(df_ht_BR['Lh'].values, [25, 50, 75])
dfLh = pd.DataFrame(data=[[DRq1, DRmed, DRq3, df_ht_DR['Lh'].mean()], [BRq1, BRmed, BRq3, df_ht_BR['Lh'].mean()]], 
                    columns=['q25','q50','q75', 'mean'], index=['DR','BR'])
dfLh.to_csv(path + 'df_Lh_stats.csv', float_format="%.1f")
axs[0].vlines(pos, [DRq1, BRq1], [DRq3, BRq3], color='k', linestyle='-', lw=5)
axs[0].set_ylim((-10,800))
axs[0].set_xticks(pos)
axs[0].set_xticklabels(label)
axs[0].set_ylabel('Length (m)')
axs[0].set_title('Hillslope Length')

parts = axs[1].violinplot(R, pos, vert=True, showmeans=False, showmedians=True,
        showextrema=True)
for pc, color in zip(parts['bodies'], clrs):
    pc.set_facecolor(color)
    pc.set_alpha(0.8)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp = parts[partname]
    vp.set_edgecolor("k")
    vp.set_linewidth(1)
DRq1, DRmed, DRq3 = np.percentile(df_ht_DR['R'].values, [25, 50, 75])
BRq1, BRmed, BRq3 = np.percentile(df_ht_BR['R'].values, [25, 50, 75])
dfR = pd.DataFrame(data=[[DRq1, DRmed, DRq3, df_ht_DR['R'].mean()], [BRq1, BRmed, BRq3, df_ht_BR['R'].mean()]], 
                    columns=['q25','q50','q75', 'mean'], index=['DR','BR'])
dfR.to_csv(path + 'df_Relief_stats.csv', float_format="%.1f")
axs[1].vlines(pos, [DRq1, BRq1], [DRq3, BRq3], color='k', linestyle='-', lw=5)

axs[1].set_xticks(pos)
axs[1].set_xticklabels(label)
axs[1].set_title('Hillslope Relief')
axs[1].set_ylabel('Height (m)')
plt.show()
fig.tight_layout()
plt.savefig(figpath+'Lh_R_violinplot.png')
plt.savefig(figpath+'Lh_R_violinplot.pdf')

#%% Violin plots of ridgetop curvature and E*

# drop some bad points with very large negative curvatures
cBR = df_ht_BR['Cht'] > -1
cDR = df_ht_DR['Cht'] > -1

pos   = [1, 2]
label = ['Druids Run', 'Baisman Run']
Cht = [np.log10(-df_ht_DR['Cht'][cDR]), np.log10(-df_ht_BR['Cht'][cBR])]
Estar = [np.log10(df_ht_DR['E_Star'][cDR]), np.log10(df_ht_BR['E_Star'][cBR])]

fig, axs = plt.subplots(ncols=2, figsize=(8,5))
parts = axs[0].violinplot(Cht, pos, vert=True, showmeans=False, showmedians=True,
        showextrema=True)
for pc, color in zip(parts['bodies'], clrs):
    pc.set_facecolor(color)
    pc.set_alpha(0.8)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp = parts[partname]
    vp.set_edgecolor("k")
    vp.set_linewidth(1)
DRq1, DRmed, DRq3 = np.percentile(df_ht_DR['Cht'][cDR], [25, 50, 75])
BRq1, BRmed, BRq3 = np.percentile(df_ht_BR['Cht'][cBR], [25, 50, 75])
dfCht = pd.DataFrame(data=[[DRq1, DRmed, DRq3, df_ht_DR['Cht'][cDR].mean()], [BRq1, BRmed, BRq3, df_ht_BR['Cht'][cBR].mean()]], 
                    columns=['q25','q50','q75', 'mean'], index=['DR','BR'])
dfCht.to_csv(path + 'df_Cht_stats.csv', float_format="%.3e")
axs[0].vlines(pos, np.log10(-np.array([DRq1, BRq1])), np.log10(-np.array([DRq3, BRq3])), color='k', linestyle='-', lw=5)
# axs[0].set_ylim((-10,1200))
axs[0].set_xticks(pos)
axs[0].set_xticklabels(label)
axs[0].set_ylabel(r'$log_{10} (-C_{ht})$ (1/m)')
axs[0].set_title('Ridgetop Curvature')

parts = axs[1].violinplot(Estar, pos, vert=True, showmeans=False, showmedians=True,
        showextrema=True)
for pc, color in zip(parts['bodies'], clrs):
    pc.set_facecolor(color)
    pc.set_alpha(0.8)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp = parts[partname]
    vp.set_edgecolor("k")
    vp.set_linewidth(1)
DRq1, DRmed, DRq3, DR95, DR99 = np.percentile(df_ht_DR['E_Star'][cDR], [25, 50, 75, 95, 99])
BRq1, BRmed, BRq3, BR95, BR99 = np.percentile(df_ht_BR['E_Star'][cBR], [25, 50, 75, 95, 99])
dfR = pd.DataFrame(data=[[DRmed, DR95, DR99, df_ht_DR['E_Star'][cDR].mean()], [BRmed, BR95, BR99, df_ht_BR['E_Star'][cBR].mean()]], 
                    columns=['q50','q95','q99', 'mean'], index=['DR','BR'])
dfR.to_csv(path + 'df_estar_stats.csv', float_format="%.1f")
axs[1].vlines(pos, np.log10(np.array([DRq1, BRq1])), np.log10(np.array([DRq3, BRq3])), color='k', linestyle='-', lw=5)
axs[1].axhline(y=0, linestyle='--', color='k', label='E*=1')
axs[1].set_xticks(pos)
axs[1].set_xticklabels(label)
axs[1].set_title('Dimensionless Erosion Rate')
axs[1].set_ylabel(r'$log_{10} (E^*)$ (-)')
plt.show()
fig.tight_layout()
plt.savefig(figpath+'Cht_estar_violinplot.png')
plt.savefig(figpath+'Cht_estar_violinplot.pdf')

#%% plot correlation between Lh and Cht


cht_binned_DR, lh_binned_DR, bin_no = binned_statistic(df_ht_DR['Lh'], df_ht_DR['Cht'], statistic='median', bins=np.quantile(df_ht_DR['Lh'], np.linspace(0,1,20)))
cht_binned_BR, lh_binned_BR, bin_no = binned_statistic(df_ht_BR['Lh'], df_ht_BR['Cht'], statistic='median', bins=np.quantile(df_ht_BR['Lh'], np.linspace(0,1,20)))
fig, axs = plt.subplots(ncols=2, figsize=(6,3))
axs[0].scatter(df_ht_DR['Lh'], df_ht_DR['Cht'], alpha=0.05, s=3)
axs[0].plot(np.diff(lh_binned_DR)/2+lh_binned_DR[:-1], cht_binned_DR, 'k--')
axs[1].scatter(df_ht_BR['Lh'], df_ht_BR['Cht'], alpha=0.05, s=3)
axs[1].plot(np.diff(lh_binned_BR)/2+lh_binned_BR[:-1], cht_binned_BR, 'k--')
axs[0].set_ylim((-0.02, 0.001))
axs[1].set_ylim((-0.02, 0.001))
axs[0].set_ylabel(r'$C_{ht}$ (1/m)')
axs[0].set_xlabel(r'$L_h$ (m)')
axs[1].set_xlabel(r'$L_h$ (m)')
axs[0].set_title('Druids Run')
axs[1].set_title('Baisman Run')
fig.tight_layout()
plt.savefig(figpath+'Cht_Lh_scatter.png')
plt.savefig(figpath+'Cht_Lh_scatter.pdf')

# %% plot basin

# Druids Run
basin_name = 'baltimore2015_DR1_AllBasins.bil'
bsn = rd.open(path_DR + basin_name)
basin = bsn.read(1) > 0 

# Baisman Run
basin_name = 'baltimore2015_BR_AllBasins.bil'
bsn = rd.open(path_BR + basin_name)
basin = bsn.read(1) > 0 

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

conc = '0.5'

# name_chi_DR = "baltimore2015_DR1_chi_data_map.csv"
# name_chi_BR = "baltimore2015_BR_chi_data_map.csv"

name_chi_DR = "baltimore2015_DR1_%s_MChiSegmented.csv"%conc
name_chi_BR = "baltimore2015_BR_%s_MChiSegmented.csv"%conc

df_chi_DR = pd.read_csv(path_DR + name_chi_DR)
df_chi_BR = pd.read_csv(path_BR + name_chi_BR)

#%% chi-elevation for drainage area larger than threshold

Quant_DR = np.quantile(df_chi_DR['drainage_area'], 0.4)
Quant_BR = np.quantile(df_chi_BR['drainage_area'], 0.4)
sel_DR = df_chi_DR['drainage_area']>Quant_DR
sel_BR = df_chi_BR['drainage_area']>Quant_BR


df_chi_DR1 = df_chi_DR.loc[df_chi_DR['drainage_area']>Quant_DR]
df_chi_BR1 = df_chi_BR.loc[df_chi_BR['drainage_area']>Quant_BR]

fig, axs = plt.subplots(ncols=2, figsize=(7,3.5))
sc = axs[0].scatter(df_chi_DR1['chi'], df_chi_DR1['elevation'], c=np.log10(df_chi_DR1['m_chi']), s=3, zorder=99)
axs[0].scatter(df_chi_DR['chi'], df_chi_DR['elevation'], c='0.8', s=3, zorder=90)
axs[0].set_xlabel(r'$\chi$ (m)')
axs[0].set_ylabel('Elevation (m)')
axs[0].set_title('Druids Run')
plt.colorbar(sc, label=r'log$_{10}$( $k_{sn}$)', ax=axs[0])

sc = axs[1].scatter(df_chi_BR1['chi'], df_chi_BR1['elevation'], c=np.log10(df_chi_BR1['m_chi']), s=3, zorder=99)
axs[1].scatter(df_chi_BR['chi'], df_chi_BR['elevation'], c='0.8', s=3, zorder=90)
axs[1].set_xlabel(r'$\chi$ (m)')
axs[1].set_ylabel('Elevation (m)')
axs[1].set_title('Baisman Run')
plt.colorbar(sc, label=r'log$_{10}$( $k_{sn}$)', ax=axs[1])
plt.tight_layout()
plt.savefig(figpath+'chi_BR_DR_%s.png'%conc, dpi=300, transparent=True)
plt.savefig(figpath+'chi_BR_DR_%s.pdf'%conc, transparent=True)


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

plt.figure()
df_chi_BR['m_chi'].plot.density(color='r', label='Baisman')
df_chi_DR['m_chi'].plot.density(color='b', label='Druids')
plt.xlabel('log10 $k_{sn}$')
plt.legend(frameon=False)
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/ksn_segment_distr.png')


plt.figure()
df_chi_BR['drainage_area'].plot.cdf(color='r', label='Baisman')
df_chi_DR['drainage_area'].plot.density(color='b', label='Druids')
# plt.xlabel('log10 $k_{sn}$')
plt.legend(frameon=False)

# %% Chi-elevation plots

fig, axs = plt.subplots()
sc = axs.scatter(df_chi_DR['chi'], df_chi_DR['elevation'], c=df_chi_DR['m_chi'], s=3)
axs.set_xlabel(r'$\chi$ (m)')
axs.set_ylabel('Elevation (m)')
axs.set_title('Druids Run: m/n=%s'%conc)
fig.colorbar(sc, label=r'log10 $k_{sn}$')
# plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/chi_DR_%s.png'%conc)

fig, axs = plt.subplots()
sc = axs.scatter(df_chi_BR['chi'], df_chi_BR['elevation'], c=df_chi_BR['m_chi'], s=3)
axs.set_xlabel(r'$\chi$ (m)')
axs.set_ylabel('Elevation')
axs.set_title('Baisman Run: m/n=%s'%conc)
fig.colorbar(sc, label=r'log10 $k_{sn}$')
# plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/chi_BR_%s.png'%conc)

# %% map Ksn

# name = 'baltimore2015_DR1_hs.bil' # hillshade
# df = df_chi_DR
# src = rd.open(path_DR + name) # hillshade

name = 'baltimore2015_BR_hs.bil'
df = df_chi_BR
src = rd.open(path_BR + name) # hillshade

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
# plt.savefig(figpath+'DruidRun_ksn.png')
plt.savefig(figpath+'OregonRidge_ksn.png')
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
# plt.savefig(figpath+'DruidRun_chi.png')
plt.savefig(figpath+'OregonRidge_chi.png')
plt.show()
# %%
