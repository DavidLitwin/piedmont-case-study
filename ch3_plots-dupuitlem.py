
#%%

import glob
import numpy as np
import pandas as pd
import copy
import linecache

from matplotlib import cm, colors, ticker
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LightSource
from landlab.io.netcdf import from_netcdf
plt.rc('text', usetex=True)

from generate_colormap import get_continuous_cmap


directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
base_output_path = 'CaseStudy_16'
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

plt.figure(figsize=(6,5))
for ID in model_runs:
    df_r_change = pd.read_csv('%s/%s/relief_change_%d.csv'%(directory, base_output_path,ID))    
    plt.plot(df_r_change['t_nd'][1:], df_r_change['r_nd'][1:]) # - df_r_change['r_nd'][1]

# plt.xlim((-50,2050))
# plt.ylim((-50,1250))
plt.legend(frameon=False)
plt.xlabel(r'$t/t_g$', fontsize=14)
plt.ylabel(r'$\bar{z} / h_g$', fontsize=14)
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

axs[-1, 0].set_ylabel(r'$y$ (m)')
axs[-1, 0].set_xlabel(r'$x$ (m)')
# plt.subplots_adjust(left=0.15, bottom=0.15, right=None, top=None, wspace=0.15, hspace=0.15)
plt.savefig('%s/%s/hillshade_%s.png'%(directory, base_output_path, base_output_path), dpi=300)

#%% Saturation class

# colorbar approach courtesy of https://stackoverflow.com/a/53361072/11627361, https://stackoverflow.com/a/60870122/11627361, 

# event
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6,5)) #(8,5)
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    sat_storm = grid.at_node['sat_mean_end_storm']
    sat_interstorm = grid.at_node['sat_mean_end_interstorm']
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
plt.savefig('%s/%s/sat_zones_%s.png'%(directory, base_output_path, base_output_path), dpi=300)

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





# %%
#%% saturation discharge 

cmap1 = copy.copy(cm.viridis)
cmap1.set_bad(cmap1(0))

i_max = 0

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,6)) #(8,6)

for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    # df = pickle.load(open('%s/%s/q_s_dt_ID_%d.p'%(directory, base_output_path, i), 'rb'))
    df = pd.read_csv('%s/%s/q_s_dt_ID_%d.csv'%(directory, base_output_path, i))
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))

    Atot = np.sum(grid.cell_area_at_node[grid.core_nodes])
    Qb = df['qb']/(Atot*df_params['p'][i])
    
    Q = df['qs_star']
    S = df['S_star']
    qs_cells = df['sat_nodes']/grid.number_of_cells #same as number of core nodes
    r = df['r']
    ibar = (df_params['ds'][i]/df_params['tr'][i])*np.sum(grid.cell_area_at_node)
    i_max = max(i_max, max(r/ibar))
    rstar = r/ibar
    
    sort = np.argsort(S)
    ind = 1000

    sc = axs[m, n].scatter(Q[ind:], 
                            qs_cells[ind:], 
                            color='lightgray', 
                            s=4, 
                            alpha=0.05,
                            rasterized=True)
    
    sc = axs[m, n].scatter(Qb[ind:], #[sort[ind:]] 
                            qs_cells[ind:], 
                            c=S[ind:], 
                            s=4, 
                            alpha=0.2, 
                            # vmin=0.0,
                            # vmax=1.0,
                            norm=colors.LogNorm(vmin=1e-3, vmax=1), 
                            cmap=cmap1,
                            rasterized=True)

    axs[m, n].text(0.05, 
                    0.92, 
                    str(i), 
                    transform=axs[m, n].transAxes, 
                    fontsize=8, 
                    verticalalignment='top',
                    )
    axs[m, n].ticklabel_format(style='sci')
    axs[m, n].tick_params(axis='both', which='major')
    # axs[m ,n].set_ylim((1e-3,1))
    # axs[m ,n].set_xlim((1e-4,200))
    axs[m ,n].set_yscale('log')
    axs[m ,n].set_xscale('log')
    # if m != nrows-1:
    #     axs[m, n].set_xticklabels([])
    # if n != 0:
    #     axs[m, n].set_yticklabels([])

fig.subplots_adjust(right=0.75, hspace=0.15, wspace=0.15)
rect_cb = [0.8, 0.35, 0.03, 0.3]
ax_cb = plt.axes(rect_cb)
cbar = fig.colorbar(sc, cax=ax_cb, orientation="vertical", extend="min")
cbar.set_label(label='$S^*$', size=16)
cbar.solids.set_edgecolor("face")

axs[-1, 0].set_ylabel('$A^*_{sat}$', size=16)
axs[-1, 0].set_xlabel('$Q_b^*$', size=16)


# plt.savefig('%s/%s/Q_sat_S_%s.png'%(directory, base_output_path), dpi=300)
# plt.savefig('%s/%s/Qb_sat_S.pdf'%(directory, base_output_path), dpi=300)
# plt.close()

# %%


inds = [10, 22]

i=0

fig, axs = plt.subplots(figsize=(4,5))


grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
sat_storm = grid.at_node['sat_mean_end_storm']
sat_interstorm = grid.at_node['sat_mean_end_interstorm']
sat_class = grid.at_node['saturation_class']
labels = ["dry", "variable", "wet"]    

dx = grid.dx
y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5

L1 = ["peru", "dodgerblue", "navy"]
cmap = colors.ListedColormap(L1)
norm = colors.BoundaryNorm(np.arange(-0.5, 3), cmap.N)
fmt = ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

im = axs.imshow(sat_class.reshape(grid.shape).T, 
                        origin="lower", 
                        extent=(x[0], x[-1], y[0], y[-1]), 
                        cmap=cmap,
                        norm=norm,
                        interpolation="none",
                        )
axs.axis('off')
plt.savefig('%s/%s/sat_class_%s_%d.png'%(directory, base_output_path, base_output_path,i), dpi=300, transparent=True)

# %%
