
#%%

import os 
import glob
import numpy as np
import pandas as pd
import copy
import linecache
import rasterio as rd
import cartopy.crs as ccrs

from matplotlib import cm, colors, ticker
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LightSource
from landlab.io.netcdf import from_netcdf
# plt.rc('text', usetex=True)

from mpl_point_clicker import clicker
from generate_colormap import get_continuous_cmap

# directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
directory = '/Users/dlitwin/Documents/Research Data/HPC output/DupuitLEMResults/post_proc/'
base_output_path = 'CaseStudy_cross_6'
model_runs = np.arange(4)
nrows = 2
ncols = 2

#%% load results and parameters

dfs = []
for ID in model_runs:
    try:
        df = pd.read_csv('%s/%s/output_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
    except FileNotFoundError:
        df =  pd.DataFrame(columns=df.columns)
    dfs.append(df)
df_results = pd.concat(dfs, axis=1, ignore_index=True).T

dfs = []
for ID in model_runs:
    df = pd.read_csv('%s/%s/params_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
    dfs.append(df)
df_params = pd.concat(dfs, axis=1, ignore_index=True).T
df_params.to_csv('%s/%s/params.csv'%(directory,base_output_path), index=True, float_format='%.3e')

plot_runs = model_runs
df_params['label'] = ['DR-DR', 'BR-BR', 'DR-BR', 'BR-DR']
plot_array = np.array([[0, 3],
                       [2, 1]])
# plot_runs = model_runs
# plot_array = np.flipud(plot_runs.reshape((nrows, ncols))) # note flipped!


#%%

plt.figure(figsize=(5,3))
for ID in model_runs:
    df_r_change = pd.read_csv('%s/%s/relief_change_%d.csv'%(directory, base_output_path,ID))    
    plt.plot(df_r_change['t_nd'][1:]*df_params['tg'].loc[ID]*1/(3600*24*365), 
             df_r_change['r_nd'][1:]*df_params['hg'].loc[ID], 
             label=df_params['label'].loc[ID])
plt.legend(frameon=False)
# plt.xlabel(r'$t/t_g$')
# plt.ylabel(r'$\bar{z} / h_g$')
plt.xlabel(r'$t$')
plt.ylabel(r'$\bar{z}$')
plt.tight_layout()
plt.savefig('%s/%s/r_change.pdf'%(directory, base_output_path), transparent=True, dpi=300)


#%% plot_runs hillshades 

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6,6)) #(8,6)
for i in plot_runs:

    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    # grid = read_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation']
    dx = grid.dx
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 
    ls = LightSource(azdeg=135, altdeg=45)
    axs[m,n].imshow(
                    ls.hillshade(elev.reshape(grid.shape).T, 
                                vert_exag=1, 
                                dx=dx, 
                                dy=dx), 
                    origin="lower", 
                    extent=(x[0], x[-1], y[0], y[-1]), 
                    cmap='gray',
                    )
    axs[m,n].text(0.04, 
                0.95, 
                df_params['label'][i], #i, #
                transform=axs[m,n].transAxes, 
                fontsize=12, 
                verticalalignment='top',
                color='k',
                bbox=dict(ec='w',
                          fc='w', 
                          alpha=0.7,
                          boxstyle="Square, pad=0.1",
                          )
                )   
    if m != nrows-1:
        axs[m, n].set_xticklabels([])
    if n != 0:
        axs[m, n].set_yticklabels([])
    axs[m, n].set_xlim((x[0],x[-1]))
    axs[m, n].set_ylim((y[0],y[-1]))

axs[-1, 0].set_ylabel(r'$y$ (m)')
axs[-1, 0].set_xlabel(r'$x$ (m)')


#%% plot_runs hillshades with channels

# channel network dummy figure
fig_p = plt.figure(figsize=(9,7))
ax_p = fig_p.add_subplot(1, 1, 1, projection=ccrs.UTM(18))
name = '%s-%d_pad.bil'%(base_output_path, 0) 
src = rd.open(os.path.join(directory,base_output_path,name))
bounds = src.bounds

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6,6)) #(8,6)
for i in plot_runs:

    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    # grid = read_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation']
    dx = grid.dx
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5

    # channels in model grid coordinates
    # name_ch = "%s-%d_pad_FromCHF_CN.csv"%(base_output_path, i)  # channels
    name_ch = "%s-%d_pad_AT_CN.csv"%(base_output_path, i)  # channels
    df = pd.read_csv('%s/%s/%s'%(directory, base_output_path, name_ch)) # channels
    projected_coords = ax_p.projection.transform_points(ccrs.Geodetic(), df['longitude'], df['latitude'])
    coords = projected_coords[:,0:2]
    coords_T = np.zeros_like(coords)
    coords_T[:,0] = np.max(y) - (coords[:,1] - (bounds.bottom+5*dx)) -0.5*dx# add 5*dx for the padding we addded
    coords_T[:,1] = coords[:,0] - (bounds.left) + 0.5*dx
 
    ls = LightSource(azdeg=135, altdeg=45)
    axs[m,n].imshow(
                    ls.hillshade(elev.reshape(grid.shape).T, 
                                vert_exag=1, 
                                dx=dx, 
                                dy=dx), 
                    origin="lower", 
                    extent=(x[0], x[-1], y[0], y[-1]), 
                    cmap='gray',
                    )
    axs[m,n].scatter(coords_T[:,0], coords_T[:,1], s=0.5, c='b', alpha=0.5)
    axs[m,n].text(0.04, 
                0.95, 
                df_params['label'][i], #i, #
                transform=axs[m,n].transAxes, 
                fontsize=12, 
                verticalalignment='top',
                color='k',
                bbox=dict(ec='w',
                          fc='w', 
                          alpha=0.7,
                          boxstyle="Square, pad=0.1",
                          )
                )   
    if m != nrows-1:
        axs[m, n].set_xticklabels([])
    if n != 0:
        axs[m, n].set_yticklabels([])
    axs[m, n].set_xlim((x[0],x[-1]))
    axs[m, n].set_ylim((y[0],y[-1]))

axs[-1, 0].set_ylabel(r'$y$ (m)')
axs[-1, 0].set_xlabel(r'$x$ (m)')

plt.savefig('%s/%s/hillshade_%s.png'%(directory, base_output_path, base_output_path), dpi=300, transparent=True)
plt.savefig('%s/%s/hillshade_%s.pdf'%(directory, base_output_path, base_output_path), transparent=True)


#%% Saturation class

# colorbar approach courtesy of https://stackoverflow.com/a/53361072/11627361, https://stackoverflow.com/a/60870122/11627361, 

# event
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6,5)) #(8,5)
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    sat_class = grid.at_node['saturation_class']
    labels = ["dry", "variable", "wet"]    
    
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
    dx = grid.dx/df_params['lg'][i]
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 
    L1 = ["peru", "dodgerblue", "navy"]
    L2 = ['r', 'g', 'b']
    cmap = colors.ListedColormap(L1)
    norm = colors.BoundaryNorm(np.arange(-0.5, 3), cmap.N)
    fmt = ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    im = axs[m, n].imshow(sat_class.reshape(grid.shape).T, 
                         origin="lower", 
                         extent=(x[0], x[-1], y[0], y[-1]), 
                         cmap=cmap,
                         norm=norm,
                         interpolation="none",
                         )
    axs[m,n].text(0.04, 
                0.95, 
                df_params['label'][i], # i, #
                transform=axs[m,n].transAxes, 
                fontsize=12, 
                verticalalignment='top',
                color='k',
                bbox=dict(ec='w',
                          fc='w', 
                          alpha=0.7,
                          boxstyle="Square, pad=0.1",
                          )
                )   
    axs[m, n].axis('off')
    
plt.subplots_adjust(left=0.15, right=0.8, wspace=0.05, hspace=0.05)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7]) #Left, Bottom, Width, Height
fig.colorbar(im, cax=cbar_ax, format=fmt, ticks=np.arange(0,3))
plt.savefig('%s/%s/sat_zones_%s.png'%(directory, base_output_path, base_output_path), dpi=300, transparent=True)
plt.savefig('%s/%s/sat_zones_%s.pdf'%(directory, base_output_path, base_output_path), transparent=True)


#%% sat class compare

# load predicted saturation from script
# path = 'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/'
path = '/Users/dlitwin/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/'
DR_name, BR_name = 'df_sat_DR.csv', 'df_sat_BR.csv'

df_1 = pd.read_csv(path+DR_name, names=['class', 'Druids Run'], header=0, index_col='class').T
df_2 = pd.read_csv(path+BR_name, names=['class', 'Baisman Run'], header=0, index_col='class').T

df_3 = df_results[['sat_never', 'sat_variable', 'sat_always']]
df_3['index'] = ['DR-DR', 'BR-BR', 'DR-BR', 'BR-DR']
df_3.set_index('index', inplace=True)

df_satall = pd.concat([df_1, df_2, df_3], axis=0)
df_satall = df_satall.reindex(['Druids Run', 'DR-DR', 'DR-BR', 'Baisman Run', 'BR-BR', 'BR-DR'])

# make figure
fig, ax = plt.subplots(figsize=(5,3.5))
bottom = np.zeros(6)
width=0.5 

color_dict = {'sat_never':"peru", 'sat_variable':"dodgerblue", 'sat_always':"navy"}

pos = [1,2,3,4.5,5.5,6.5]
for col in df_satall.columns:
    p = ax.bar(pos, 
               df_satall[col].values, 
               width, 
               label=col.split('_')[-1], 
               bottom=bottom,
               color=color_dict[col],
               )
    bottom += df_satall[col].values
ax.set_ylim((0,1))
ax.set_ylabel('Fractional Area')
ax.set_xticks(pos)
ax.set_xticklabels(df_satall.index, ha='right', rotation=45)
## fig.legend(loc='outside center right', frameon=False)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
# ax.legend(bbox_to_anchor=(1.0, 0.6),
#           bbox_transform=fig.transFigure,
#           frameon=False, 
#           )
ax.spines[['right', 'top']].set_visible(False)
plt.tight_layout()
plt.show()
plt.savefig('%s/%s/sat_compare_%s.png'%(directory, base_output_path, base_output_path), dpi=300, transparent=True)
plt.savefig('%s/%s/sat_compare_%s.pdf'%(directory, base_output_path, base_output_path), transparent=True)

#%% Hillshades (projected coordinates) - define channel network

i = 3
path = directory + f'/{base_output_path}/'
name_elev = '%s-%d_pad.bil'%(base_output_path, i) # elevation
src_elev = rd.open(path + name_elev) # elevation
bounds = src_elev.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
proj = src_elev.crs
utm = 18

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.UTM(utm))
ls = LightSource(azdeg=135, altdeg=45)
klicker = clicker(ax, ["source"], markers=["o"])
ax.set_extent(Extent, crs=ccrs.UTM(utm))
cs = ax.imshow(
                ls.hillshade(src_elev.read(1), 
                             vert_exag=1),
                cmap='gray', 
                #vmin=100,
                extent=Extent, 
                transform=ccrs.UTM(utm), 
                origin="upper")
plt.show()

#%% get points, save output

out = klicker.get_positions()
pts_utm = out['source']
geo = ccrs.Geodetic()
pts_latlon = geo.transform_points(ccrs.UTM(utm), pts_utm[:,0], pts_utm[:,1])

df_sources = pd.DataFrame({'x':pts_utm[:,0], 
                        'y':pts_utm[:,1], 
                        'latitude':pts_latlon[:,1], 
                        'longitude':pts_latlon[:,0]})
df_sources.to_csv('%s/%s/%s-%d_EyeSources.csv'%(directory, base_output_path, base_output_path, i))

#%% Hillslope length and relief: model 1


files_ht = ["%s-%d_pad_HilltopData_TN.csv"%(base_output_path, i) for i in range(4)]
df_ht_all = [pd.read_csv(os.path.join(directory,base_output_path,file_ht)) for file_ht in files_ht]
names = ['DR-DR', 'BR-BR', 'DR-BR', 'BR-DR'] # in order

#%% Hillslope length and relief: model 2

base_output_path_2 = 'CaseStudy_cross_1'
files_ht_2 = ["%s-%d_pad_HilltopData_TN.csv"%(base_output_path_2, i) for i in range(4)]
df_ht_all_2 = [pd.read_csv(os.path.join(directory,base_output_path_2,file_ht)) for file_ht in files_ht_2]
names_2 = ['DR-DR', 'BR-BR', 'DR-BR', 'BR-DR'] # in order

#%% Hillslope length and relief: field

# path1 = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/LSDTT/'
# path2 = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/LSDTT/'
path1 = '/Users/dlitwin/Documents/Research/Soldiers Delight/data/LSDTT/'
path2 = '/Users/dlitwin/Documents/Research/Oregon Ridge/data/LSDTT/'

name_ht_DR = "baltimore2015_DR1_HilltopData_TN.csv"
name_ht_BR = "baltimore2015_BR_HilltopData_TN.csv"

df_ht_DR = pd.read_csv(path1 + name_ht_DR)
df_ht_DR = df_ht_DR[df_ht_DR['BasinID']==99]
df_ht_BR = pd.read_csv(path2 + name_ht_BR)
df_ht_BR = df_ht_BR[df_ht_BR['BasinID']==71]

df_ht_sites = [df_ht_DR, df_ht_BR]
names_sites = ['Druids Run', 'Baisman Run']

#%% collect sites and models

# collect things for main model results
names += names_sites
df_ht_all += df_ht_sites
ht_dict = dict(zip(names, df_ht_all))

# collect things for alternate model results
names_2 += names_sites
df_ht_all_2 += df_ht_sites
ht_dict_2 = dict(zip(names_2, df_ht_all_2))

#%% violin plots of hillslope length and relief

# select whether we want to plot original or alt
ht_dict_plot = ht_dict #ht_dict_2 #

# select order and set some plot attributes
labels = ['Druids Run', 'DR-DR', 'DR-BR', 'Baisman Run', 'BR-BR', 'BR-DR']
clrs = ['firebrick', 'gray', 'gray', 'royalblue','gray', 'gray']
pos   = [1, 2, 3, 4.5, 5.5, 6.5]
alph = [0.8, 0.4, 0.4, 0.8, 0.4, 0.4]

# get them in the right order and then get the values we want
dfs = [ht_dict_plot[label] for label in labels]
Lh = [df['Lh'].values for df in dfs]
R = [df['R'].values for df in dfs]
Cht = [np.log10(-df['Cht'][df['Cht']>-500].values) for df in dfs]


#%% hillslope length and relief violin plots 

fig, axs = plt.subplots(ncols=2, figsize=(6,4))
parts = axs[0].violinplot(Lh, pos, vert=True, showmeans=False, showmedians=True,
        showextrema=True)
for pc, color in zip(parts['bodies'], clrs):
    pc.set_facecolor(color)
    pc.set_alpha(0.8)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp = parts[partname]
    vp.set_edgecolor("k")
    vp.set_linewidth(1)

q1s = [np.percentile(lh, 25) for lh in Lh]
q3s = [np.percentile(lh, 75) for lh in Lh]
meds = [np.percentile(lh, 50) for lh in Lh]
# dfLh = pd.DataFrame(data=[[DRq1, DRmed, DRq3, df_ht_DR['Lh'].mean()], [BRq1, BRmed, BRq3, df_ht_BR['Lh'].mean()]], 
#                     columns=['q25','q50','q75', 'mean'], index=['DR','BR'])
# dfLh.to_csv('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/df_Lh_stats.csv', float_format="%.1f")
axs[0].vlines(pos, q1s, q3s, color='k', linestyle='-', lw=3)
axs[0].set_xticks(pos)
axs[0].set_xticklabels(labels, rotation=45, ha='right')
axs[0].set_yscale('log')
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
q1s = [np.percentile(r, 25) for r in R]
q3s = [np.percentile(r, 75) for r in R]
meds = [np.percentile(r, 50) for r in R]
# dfR = pd.DataFrame(data=[[DRq1, DRmed, DRq3, df_ht_DR['R'].mean()], [BRq1, BRmed, BRq3, df_ht_BR['R'].mean()]], 
#                     columns=['q25','q50','q75', 'mean'], index=['DR','BR'])
# dfR.to_csv('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/df_R_modeled_stats.csv', float_format="%.1f")
axs[1].vlines(pos, q1s, q3s, color='k', linestyle='-', lw=3)
axs[1].set_xticks(pos)
axs[1].set_xticklabels(labels, rotation=45, ha='right')
axs[1].set_yscale('log')
axs[1].set_title('Hillslope Relief')
axs[1].set_ylabel('Height (m)')
plt.show()
fig.tight_layout()
plt.savefig('%s/%s/Lh_R_violinplot_%s.png'%(directory, base_output_path, base_output_path), dpi=300, transparent=True)
plt.savefig('%s/%s/Lh_R_violinplot_%s.pdf'%(directory, base_output_path, base_output_path), transparent=True)


#%% hilltop curvature violin plot

fig, axs = plt.subplots(figsize=(4,4))
parts = axs.violinplot(Cht, pos, vert=True, showmeans=False, showmedians=True,
        showextrema=True)
for pc, color in zip(parts['bodies'], clrs):
    pc.set_facecolor(color)
    pc.set_alpha(0.8)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp = parts[partname]
    vp.set_edgecolor("k")
    vp.set_linewidth(1)

q1s = [np.percentile(c, 25) for c in Cht]
q3s = [np.percentile(c, 75) for c in Cht]
meds = [np.percentile(c, 50) for c in Cht]
axs.vlines(pos, q1s, q3s, color='k', linestyle='-', lw=3)
axs.set_xticks(pos)
axs.set_xticklabels(labels, rotation=45, ha='right')
# axs.set_yscale('log')
axs.set_ylabel('$log_{10}(-C_{ht})$ (1/m)')
axs.set_title('Hilltop Curvature')
fig.tight_layout()
plt.savefig('%s/%s/Cht_violinplot_%s.png'%(directory, base_output_path, base_output_path), dpi=300, transparent=True)
plt.savefig('%s/%s/Cht_violinplot_%s.pdf'%(directory, base_output_path, base_output_path), transparent=True)


#%% Channels

# sites
conc = 0.5
name_chi_DR = "baltimore2015_DR1_%s_MChiSegmented.csv"%conc
name_chi_BR = "baltimore2015_BR_%s_MChiSegmented.csv"%conc

df_chi_DR = pd.read_csv(path1 + name_chi_DR)
df_chi_BR = pd.read_csv(path2 + name_chi_BR)

# Quant_DR = np.quantile(df_chi_DR['drainage_area'], 0.4)
# Quant_BR = np.quantile(df_chi_BR['drainage_area'], 0.4)

# df_chi_DR1 = df_chi_DR.loc[df_chi_DR['drainage_area']>Quant_DR]
# df_chi_BR1 = df_chi_BR.loc[df_chi_BR['drainage_area']>Quant_BR]

df_chi_sites = [df_chi_DR, df_chi_BR]
names_sites = ['Druids Run', 'Baisman Run']

# models
files_chi = ["%s-%d_pad_MChiSegmented.csv"%(base_output_path, i) for i in range(4)]
df_chi_all = [pd.read_csv(os.path.join(directory,base_output_path,file_chi)) for file_chi in files_chi]
names = ['DR-DR', 'BR-BR', 'DR-BR', 'BR-DR'] # in order

# collect things for main model results
names += names_sites
df_chi_all += df_chi_sites
chi_dict = dict(zip(names, df_chi_all))

# %% Chi analysis visualization

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6,6)) #(8,6)
for i in plot_runs:

    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]

    df_chi = df_chi_all[i]
    quant = np.quantile(df_chi['drainage_area'], 0.2)
    df_chi1 = df_chi.loc[df_chi['drainage_area']>quant]

    sc = axs[m,n].scatter(df_chi1['chi'], df_chi1['elevation'], c=df_chi1['m_chi'], s=3, zorder=99)
    axs[m,n].scatter(df_chi['chi'], df_chi['elevation'], c='0.8', s=3, zorder=90)
    axs[m,n].set_xlabel(r'$\chi$ (m)')
    axs[m,n].set_ylabel('Elevation (m)')
    axs[m,n].set_title(names[i])
    plt.colorbar(sc, label=r'log$_{10}$( $k_{sn}$)', ax=axs[m,n])
plt.tight_layout()
plt.savefig('%s/%s/chi_elevation_%s.png'%(directory, base_output_path, base_output_path), dpi=300, transparent=True)
plt.savefig('%s/%s/chi_elevation_%s.png'%(directory, base_output_path, base_output_path), transparent=True)

#%% violin plots of hillslope length and relief

# select order and set some plot attributes
labels = ['Druids Run', 'DR-DR', 'DR-BR', 'Baisman Run', 'BR-BR', 'BR-DR']
clrs = ['firebrick', 'gray', 'gray', 'royalblue','gray', 'gray']
pos   = [1, 2, 3, 4.5, 5.5, 6.5]
alph = [0.8, 0.4, 0.4, 0.8, 0.4, 0.4]

# get them in the right order and then get the values we want
dfs = [chi_dict[label] for label in labels]
ksn = [np.log10(df['m_chi'].values) for df in dfs]


fig, axs = plt.subplots(figsize=(4,4))
parts = axs.violinplot(ksn, pos, vert=True, showmeans=False, showmedians=True,
        showextrema=True)
for pc, color in zip(parts['bodies'], clrs):
    pc.set_facecolor(color)
    pc.set_alpha(0.8)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp = parts[partname]
    vp.set_edgecolor("k")
    vp.set_linewidth(1)

q1s = [np.percentile(c, 25) for c in ksn]
q3s = [np.percentile(c, 75) for c in ksn]
meds = [np.percentile(c, 50) for c in ksn]
axs.vlines(pos, q1s, q3s, color='k', linestyle='-', lw=3)
axs.set_xticks(pos)
axs.set_xticklabels(labels, rotation=45, ha='right')
# axs.set_yscale('log')
axs.set_ylabel(r'log10 $k_{sn}$')
axs.set_title('Channel Steepness')
fig.tight_layout()
plt.savefig('%s/%s/ksn_violinplot_%s.png'%(directory, base_output_path, base_output_path), dpi=300, transparent=True)
plt.savefig('%s/%s/ksn_violinplot_%s.pdf'%(directory, base_output_path, base_output_path), transparent=True)




#%% scatter with errorbar

# assemble 
# df_ht_sites = [df_ht_DR, df_ht_BR]
# names_sites = ['Druids Run', 'Baisman Run']

labels = ['DR-DR', 'BR-BR']
dfs_1 = [ht_dict[label] for label in labels]
dfs_2 = [ht_dict_2[label] for label in labels]

dfs_models = dfs_1 + dfs_2
dfs_sites = df_ht_sites + df_ht_sites

labels = ['DR-DR (1)', 'BR-BR (1)', 'DR-DR (2)', 'BR-BR (2)']
df_all = pd.DataFrame(index=labels)

df_all['Lh_q1_mod'] = [np.percentile(df['Lh'].values, 25) for df in dfs_models]
df_all['Lh_q3_mod']  = [np.percentile(df['Lh'].values, 75) for df in dfs_models]
df_all['Lh_q2_mod']  = [np.percentile(df['Lh'].values, 50) for df in dfs_models]

df_all['Lh_q1_site'] = [np.percentile(df['Lh'].values, 25) for df in dfs_sites]
df_all['Lh_q3_site']  = [np.percentile(df['Lh'].values, 75) for df in dfs_sites]
df_all['Lh_q2_site']  = [np.percentile(df['Lh'].values, 50) for df in dfs_sites]

df_all['R_q1_mod'] = [np.percentile(df['R'].values, 25) for df in dfs_models]
df_all['R_q3_mod']  = [np.percentile(df['R'].values, 75) for df in dfs_models]
df_all['R_q2_mod']  = [np.percentile(df['R'].values, 50) for df in dfs_models]

df_all['R_q1_site'] = [np.percentile(df['R'].values, 25) for df in dfs_sites]
df_all['R_q3_site']  = [np.percentile(df['R'].values, 75) for df in dfs_sites]
df_all['R_q2_site']  = [np.percentile(df['R'].values, 50) for df in dfs_sites]

df_all['Lh_site_lower'] = df_all['Lh_q2_site']-df_all['Lh_q1_site']
df_all['R_site_lower'] = df_all['R_q2_site']-df_all['R_q1_site']
df_all['Lh_site_upper'] = df_all['Lh_q3_site']-df_all['Lh_q2_site']
df_all['R_site_upper'] = df_all['R_q3_site']-df_all['R_q2_site']

df_all['Lh_mod_lower'] = df_all['Lh_q2_mod']-df_all['Lh_q1_mod']
df_all['R_mod_lower'] = df_all['R_q2_mod']-df_all['R_q1_mod']
df_all['Lh_mod_upper'] = df_all['Lh_q3_mod']-df_all['Lh_q2_mod']
df_all['R_mod_upper'] = df_all['R_q3_mod']-df_all['R_q2_mod']


#%% plot scatter

fig, axs = plt.subplots(ncols=2, figsize=(6,4))

plot_labels = ['DR-DR (0.3)', 'BR-BR (0.3)', 'DR-DR (0.6)']
for i in range(3):
    axs[0].errorbar([df_all['Lh_q2_site'].iloc[i]], [df_all['Lh_q2_mod'].iloc[i]], 
                        xerr=[[df_all['Lh_site_lower'].iloc[i]], [df_all['Lh_site_upper'].iloc[i]]],
                        yerr=[[df_all['Lh_mod_lower'].iloc[i]], [df_all['Lh_mod_upper'].iloc[i]]],
                        fmt="o",
                        label=plot_labels[i] #df_all.index.values[i]
                        )
    axs[1].errorbar([df_all['R_q2_site'].iloc[i]], [df_all['R_q2_mod'].iloc[i]],
                        xerr=[[df_all['R_site_lower'].iloc[i]], [df_all['R_site_upper'].iloc[i]]],
                        yerr=[[df_all['R_mod_lower'].iloc[i]], [df_all['R_mod_upper'].iloc[i]]],
                        fmt="o",
                        label=plot_labels[i] #df_all.index.values[i]
                        )
axs[0].axline([10,10],[100,100], color='k', linestyle='--', label='1:1')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel(r'Measured $\overline{L_h}$')
axs[0].set_ylabel(r'Modeled $\overline{L_h}$')
axs[1].axline([1,1],[10,10], color='k', linestyle='--')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel(r'Measured $\overline{R}$')
axs[1].set_ylabel(r'Modeled $\overline{R}$')
axs[0].legend(loc='lower left')
plt.tight_layout()
plt.show()
plt.savefig('%s/%s/Lh_R_scatter_%s.png'%(directory, base_output_path, base_output_path), dpi=300, transparent=True)
plt.savefig('%s/%s/Lh_R_scatter_%s.pdf'%(directory, base_output_path, base_output_path), transparent=True)


#%% get hilltops and curvature


i = 0
name_ch = "%s-%d_pad_FromCHF_CN.csv"%(base_output_path, i)  # channels
name_hds = "%s-%d_EyeSources.csv"%(base_output_path, i) # channel heads
name_rge = "%s-%d_pad_RidgeData.csv"%(base_output_path, i) # ridges h
name_ht = "%s-%d_pad_HilltopData_TN.csv"%(base_output_path, i) # hilltopData

df = pd.read_csv(os.path.join(directory,base_output_path, name_ch)) # channels
df1 = pd.read_csv(os.path.join(directory,base_output_path, name_hds)) # channel heads
df2 = pd.read_csv(os.path.join(directory,base_output_path, name_rge)) # ridges
df3 = pd.read_csv(os.path.join(directory,base_output_path, name_ht)) # ridges


#%%

ind = df3['Cht'] > -500
plt.figure()
plt.hist(df3['Cht'][ind], bins=50)
plt.axvline(x=df3['Cht'].median(), color='k')


#%% ################ Old things ######################


#%% old scatter Lh Rh


fig, axs = plt.subplots(ncols=2, figsize=(6,4))
for i, df in enumerate(dfs):
    q1, med, q3 = np.percentile(df['Lh'].values, [25, 50, 75])
    DRq1, DRmed, DRq3 = np.percentile(df_ht_DR['Lh'].values, [25, 50, 75])
    BRq1, BRmed, BRq3 = np.percentile(df_ht_BR['Lh'].values, [25, 50, 75])

    if i < 2:
        axs[0].errorbar([DRmed], [med], 
                    xerr=[[DRmed-DRq1], [DRq3-DRmed]], 
                    yerr=[[med-q1], [q3-med]],
                    fmt="o",
                    label=names[i]
                    )
    else:
          axs[0].errorbar([BRmed], [med], 
                    xerr=[[BRmed-BRq1], [BRq3-BRmed]], 
                    yerr=[[med-q1], [q3-med]],
                    fmt="o",
                    label=names[i]
                    )      

    q1, med, q3 = np.percentile(df['R'].values, [25, 50, 75])
    DRq1, DRmed, DRq3 = np.percentile(df_ht_DR['R'].values, [25, 50, 75])
    BRq1, BRmed, BRq3 = np.percentile(df_ht_BR['R'].values, [25, 50, 75])
    
    if i < 2:
        axs[1].errorbar(DRmed, med, 
                    xerr=[[DRmed-DRq1], [DRq3-DRmed]], 
                    yerr=[[med-q1], [q3-med]],
                    fmt="o",
                    label=names[i]
                    )
    else:
          axs[1].errorbar(BRmed, med, 
                    xerr=[[BRmed-BRq1], [BRq3-BRmed]], 
                    yerr=[[med-q1], [q3-med]],
                    fmt="o",
                    label=names[i]
                    )   
axs[0].axline([10,10],[100,100], color='k', linestyle='--', label='1:1')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel(r'Measured $\overline{L_h}$')
axs[0].set_ylabel(r'Modeled $\overline{L_h}$')
axs[1].axline([1,1],[10,10], color='k', linestyle='--')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel(r'Measured $\overline{R}$')
axs[1].set_ylabel(r'Modeled $\overline{R}$')
axs[0].legend(frameon=False)
plt.tight_layout()
plt.show()
# plt.savefig('%s/%s/Lh_R_scatter_%s.png'%(directory, base_output_path, base_output_path), dpi=300, transparent=True)
# plt.savefig('%s/%s/Lh_R_scatter_%s.pdf'%(directory, base_output_path, base_output_path), transparent=True)


# %%

inds = np.arange(4)

for i in inds:

    fig, axs = plt.subplots(figsize=(4,1.5))
    
    hg = df_params['hg'][i] # all the same hg and lg
    lg = df_params['lg'][i]
    b = df_params['b'][i]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation']
    base = grid.at_node['aquifer_base__elevation']
    wt_high = grid.at_node['wtrel_mean_end_storm']*b + base
    wt_low = grid.at_node['wtrel_mean_end_interstorm']*b + base
    middle_row = np.where(grid.x_of_node == np.median(grid.x_of_node))[0][1:-1]
    
    y = grid.y_of_node[middle_row]
    axs.fill_between(y,elev[middle_row],base[middle_row],facecolor=(198/256,155/256,126/256) )
    axs.fill_between(y,wt_high[middle_row],base[middle_row],facecolor=(145/256,176/256,227/256), alpha=1.0) #
    # axs.fill_between(y,wt_low[middle_row]/hg,base[middle_row]/hg,facecolor='royalblue', alpha=1.0)
    axs.fill_between(y,base[middle_row],np.zeros_like(base[middle_row]),facecolor=(111/256,111/256,111/256))
    axs.set_xlim((min(y),max(y)))
    axs.set_ylim((0,np.nanmax(elev[middle_row])*1.05))
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('%s/%s/cross_section_%s_%d.png'%(directory, base_output_path, base_output_path,i), dpi=300, transparent=True)


#%%
fig1, axs1 = plt.subplots(figsize=(4,4)) 
fig2, axs2 = plt.subplots(figsize=(4,4)) 

for i in model_runs:
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    qstar = grid.at_node['surface_water_effective__discharge']/(df_params['p'][i]*grid.at_node['drainage_area'])
    area = grid.at_node['drainage_area']
    
    ord = np.argsort(qstar)

    axs1.scatter(area[ord], qstar[ord], label=df_params['label'].loc[i], alpha=0.3, s=8)

    axs2.plot(qstar[ord], np.linspace(0,1,len(ord)), label=df_params['label'].loc[i], alpha=0.3)

axs1.set_xscale('log')
axs1.legend(frameon=False)
axs2.legend(frameon=False)

# %%
#%% Channels and Hillslopes on hillshade (projected coordinate method)

i = 1
path = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc/%s/'%base_output_path
name = '%s-%d_pad_hs.bil'%(base_output_path, i) # hillshade
# name_ch = "%s-%d_pad_D_CN.csv"%(base_output_path, i)  # channels
# name_hds = "%s-%d_pad_Dsources.csv"%(base_output_path, i) # channel heads
# name_rge = "%s-%d_pad_RidgeData.csv"%(base_output_path, i) # ridges
# name_ch = "%s-%d_pad_AT_CN.csv"%(base_output_path, i)  # channels
# name_hds = "%s-%d_pad_ATsources.csv"%(base_output_path, i) # channel heads
# name_rge = "%s-%d_pad_RidgeData.csv"%(base_output_path, i) # ridges
name_ch = "%s-%d_pad_FromCHF_CN.csv"%(base_output_path, i)  # channels
name_hds = "%s-%d_EyeSources.csv"%(base_output_path, i) # channel heads
name_rge = "%s-%d_pad_RidgeData.csv"%(base_output_path, i) # ridges

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
klicker = clicker(ax, ["source"], markers=["o"])

projected_coords = ax.projection.transform_points(ccrs.Geodetic(), df['longitude'], df['latitude'])
ax.scatter(projected_coords[:,0], projected_coords[:,1], s=0.5, c='b', transform=ccrs.UTM(utm)) #c=df['Stream Order'],

projected_coords = ax.projection.transform_points(ccrs.Geodetic(), df1['longitude'], df1['latitude'])
ax.scatter(projected_coords[:,0], projected_coords[:,1], s=3, c='r', transform=ccrs.UTM(utm)) #c=df['Stream Order'],

projected_coords = ax.projection.transform_points(ccrs.Geodetic(), df2['longitude'], df2['latitude'])
ax.scatter(projected_coords[:,0], projected_coords[:,1], s=0.5, c='gold', transform=ccrs.UTM(utm)) #c='g',

# Limit the extent of the map to a small longitude/latitude range.
ax.set_extent(Extent, crs=ccrs.UTM(utm))
# ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
cs = ax.imshow(src.read(1), cmap='binary', vmin=100, #cmap='plasma', vmin=-0.1, vmax=0.1, #
               extent=Extent, transform=ccrs.UTM(utm), origin="upper")
# plt.savefig('C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc/%s/%s-%d.pdf'%(base_output_path, base_output_path, i))
plt.show()

#%% compare fill dem from lsdtt

i = 1
path = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc/%s/'%base_output_path
name = '%s-%d_pad.bil'%(base_output_path, i)
name_fill = '%s-%d_pad_Fill.bil'%(base_output_path, i)

src = rd.open(path + name)
src_fill = rd.open(path + name_fill)
diff = src_fill.read(1) - src.read(1)

plt.figure()
plt.imshow(diff)
