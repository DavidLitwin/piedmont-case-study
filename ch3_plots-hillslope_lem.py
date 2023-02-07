

import copy
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
plt.rc('text', usetex=True)

from landlab.io.netcdf import from_netcdf

# directories
save_directory = 'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/'
directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc/paper1_archive'


#%% DupuitLEM: load data for hillslope length

# specify model runs
base_output_paths = ['steady_sp_3_17', 'steady_sp_3_16', 'steady_sp_3_15']  
model_runs = np.arange(40)

# load params
dfs = []
for base_output_path in base_output_paths:
    dfs.append(pickle.load(open('%s/%s/parameters.p'%(directory,base_output_path), 'rb')))
df_params = pd.concat(dfs, axis=0, ignore_index=True)
    

# load results
dfs_tog = []
for base_output_path in base_output_paths:
    
    dfs = []
    for ID in model_runs:
        
        d = pickle.load(open('%s/%s/output_ID_%d.p'%(directory,base_output_path, ID), 'rb'))
        df = pd.DataFrame([d])
        dfs.append(df)
    dfs_tog.append(pd.concat(dfs, axis=0, ignore_index=True))
df_results = pd.concat(dfs_tog, axis=0, ignore_index=True)

#%% plots of normalized hillslope length and relief

x_data = df_results['mean hillslope len']/df_params['lg']
y_data = df_results['mean hand']/df_params['hg']

fig, ax = plt.subplots(figsize=(5,4))
sc = ax.scatter(x_data**2, y_data, alpha=0.5, c=df_params['alpha']) 
ax.axline([0,0],[1,1], color='k', linestyle='--', label='1:1')
ax.set_xlabel(r'$(L_h/\ell_g)^2$')
ax.set_ylabel(r'$R_h/h_g$')
ax.legend(frameon=False)
fig.colorbar(sc, label=r'$\alpha$')
fig.tight_layout()
plt.savefig(save_directory+'hill_len_relief_steady_dupuitLEM.png', dpi=300)

# %% plots of characteristic curvature vs hg/lg^2

x_data = df_params['hg']/df_params['lg']**2
y_data = df_results['mean hand']/df_results['mean hillslope len']**2

fig, ax = plt.subplots(figsize=(5,4))
sc = ax.scatter(x_data, y_data, alpha=0.5, c=df_params['gam'], norm=colors.LogNorm()) 
ax.axline([0,0],[0.01,0.01], color='k', linestyle='--', label='1:1')
ax.set_xlabel(r'$h_g/\ell_g^2$')
ax.set_ylabel(r'$R_h/L_h^2$')
ax.legend(frameon=False)
fig.colorbar(sc, label=r'$\gamma$')
fig.tight_layout()
plt.savefig(save_directory+'char_curv_relief_steady_dupuitLEM.png', dpi=300)

# %%
