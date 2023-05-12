

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from landlab import  RasterModelGrid
import statsmodels.formula.api as smf
from landlab.io.netcdf import from_netcdf
from landlab.components import (
    GroundwaterDupuitPercolator,
    PrecipitationDistribution,
    )
from DupuitLEM.auxiliary_models import HydrologyEventVadoseStreamPower, SchenkVadoseModel

directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
base_output_path = 'CaseStudy_cross_2'
ID = 1

#%% load parameters and grid

df_params = pd.read_csv('%s/%s/params_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)

grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, ID))
elev = grid.at_node['topographic__elevation']
wt_rel = grid.at_node['wtrel_mean_end_interstorm']
TI8 = grid.at_node['topographic__index_D8']

Ks = df_params.loc['ksat'][0] #hydraulic conductivity [m/s]
ne = df_params.loc['ne'][0] #drainable porosity [-]
b = df_params.loc['b'][0] #characteristic depth  [m]
p = df_params.loc['p'][0] #average precipitation rate [m/s]
hg = df_params.loc['hg'][0]

pet = df_params.loc['pet'][0]
na = df_params.loc['na'][0] #plant available volumetric water content
tr = df_params.loc['tr'][0] #mean storm duration [s]
tb = df_params.loc['tb'][0] #mean interstorm duration [s]
ds = df_params.loc['ds'][0] #mean storm depth [m]
Nz = df_params.loc['Nz'][0] #number of schenk bins [-]

T_h = 100*(tr+tb) #total hydrological time [s]
sat_cond = 0.025 # distance from surface (units of hg) for saturation

# replace nans in elevation
elev[np.isnan(elev)] = b

#%% run hydrological model 

#initialize new grid
mg = RasterModelGrid(grid.shape, xy_spacing=grid.dx)
mg.set_status_at_node_on_edges(right=mg.BC_NODE_IS_CLOSED, top=mg.BC_NODE_IS_CLOSED, \
                              left=mg.BC_NODE_IS_FIXED_VALUE, bottom=mg.BC_NODE_IS_CLOSED)
z = mg.add_zeros('node', 'topographic__elevation')
z[:] = elev
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zb[:] = elev - b
zwt = mg.add_zeros('node', 'water_table__elevation')
zwt[:] = zb + b*wt_rel

gdp = GroundwaterDupuitPercolator(mg,
                                  porosity=ne,
                                  hydraulic_conductivity=Ks,
                                  regularization_f=0.01,
                                  recharge_rate=0.0,
                                  courant_coefficient=0.05,
                                  vn_coefficient = 0.05,
                                  )
pdr = PrecipitationDistribution(mg, mean_storm_duration=tr,
    mean_interstorm_duration=tb, mean_storm_depth=ds,
    total_t=T_h)
pdr.seed_generator(seedval=2)
svm = SchenkVadoseModel(
                potential_evapotranspiration_rate=pet,
                 available_water_content=na,
                 profile_depth=b,
                 num_bins=int(Nz),
                 )
svm.generate_state_from_analytical(ds, tb, random_seed=20220408)
hm = HydrologyEventVadoseStreamPower(
                                    mg,
                                    precip_generator=pdr,
                                    groundwater_model=gdp,
                                    vadose_model=svm,
                                    )

# hm.run_step()

# f = open('../post_proc/%s/dt_qs_s_%d.csv'%(base_output_path, ID), 'w')
# def write_SQ(grid, r, dt, file=f):
#     cores = grid.core_nodes
#     h = grid.at_node["aquifer__thickness"]
#     wt = grid.at_node["water_table__elevation"]
#     z = grid.at_node["topographic__elevation"]
#     sat = (z-wt) < sat_cond*hg
#     qs = grid.at_node["surface_water__specific_discharge"]
#     area = grid.cell_area_at_node

#     storage = np.sum(ne*h[cores]*area[cores])
#     qs_tot = np.sum(qs[cores]*area[cores])
#     sat_nodes = np.sum(sat[cores])
#     r_tot = np.sum(r[cores]*area[cores])

#     file.write('%f, %f, %f, %f, %f\n'%(dt, r_tot, qs_tot, storage, sat_nodes))
# gdp.callback_fun = write_SQ

hm.run_step_record_state()
# f.close()

#%% dataframe for output

df_output = {}
#%% get discharge and saturation for interstorms

# discharge
Q_all = hm.Q_all[1:,:]/mg.at_node['drainage_area'] * 3600*24 #m/d
intensity = hm.intensity[:-1]
Q_all = np.nanmax(Q_all[intensity==0.0,:], axis=1)

# saturation
wt_all = hm.wt_all[1:,:]
elev_all = np.ones(wt_all.shape)*mg.at_node['topographic__elevation']
sat_all = (elev_all-wt_all) < sat_cond*hg
sat_all = sat_all[intensity==0.0,:]

#%% extract transects for regression model

all_inds = np.arange(200*200).reshape(mg.shape)
n = np.arange(10,200,20)

xinds = list(all_inds[2:-2,n].flatten())
yinds = list(all_inds[n,2:-2].flatten())
inds = xinds + yinds

# %% reshape and make dataframe

# get inds and reshape
sat_obs = sat_all[:,inds]

TI_all = TI8[inds]
TI_all = TI_all.reshape(1,len(TI_all))
TI_obs = TI_all.repeat(len(Q_all), axis=0)

Q_all = Q_all.reshape(len(Q_all),1)
Q_obs = Q_all.repeat(len(inds), axis=1)

sat_obs = sat_obs.flatten()
TI_obs = TI_obs.flatten()
Q_obs = Q_obs.flatten()

# make dataframe
df_obs = pd.DataFrame({'sat_bin':sat_obs*1, 'logTI':np.log(TI_obs), 'logQ':np.log(Q_obs)})
df_obs['logTIQ'] = df_obs['logTI'] + df_obs['logQ']

# drop nans and infs
df_obs.replace([np.inf, -np.inf], np.nan, inplace=True)
df_obs.dropna(how='any', inplace=True) 

#%% regression model

model = smf.logit('sat_bin ~ logTIQ', data=df_obs).fit()

# check model performance
print(model.summary())

# predict in sample
in_sample = pd.DataFrame({'prob':model.predict()})

#%% show model prediction with data

logTIQ = np.linspace(-4,14,100)
pred = model.predict(exog=dict(logTIQ=logTIQ))

fig, ax = plt.subplots()
# ax.scatter(df_obs['logTI'], df_obs['sat_bin'], c=np.exp(df_obs['logQ']), alpha=0.3)
sc = ax.scatter(df_obs['logTIQ'], 
           df_obs['sat_bin'] + 0.05*np.random.randn(len(df_obs)), 
           c=np.exp(df_obs['logQ']),
           s=3,
           rasterized=True)
ax.plot(logTIQ, pred, 'r-')
ax.set_yticks([0,1])
ax.set_yticklabels(['N','Y'])
ax.set_ylabel('Saturated')
ax.set_xlabel('log(TIQ)')
fig.colorbar(sc, label='Q (mm/d)')
plt.savefig('%s/%s/TI_regression_%d.pdf'%(directory, base_output_path, ID), transparent=True, dpi=300)
plt.savefig('%s/%s/TI_regression_%d.png'%(directory, base_output_path, ID), transparent=True, dpi=300)


#%% calculate transmissivity: range of thresholds

N = 200
# p_all = np.linspace(0.55,0.99,N) #DR
p_all = np.geomspace(0.001,0.7,N) #BR
rhostar = lambda p: np.log(p/(1-p))
Tmean = lambda b0, b1, p: np.exp((rhostar(p)-b0)/b1)

T_all = np.zeros((N,3))
sens = np.zeros(N)
spec = np.zeros(N)
dist = np.zeros(N)

covs = model.cov_params()
means = model.params

samples = np.random.multivariate_normal(means, covs, size=10000)

for i, p in enumerate(p_all):

    Tcalc = Tmean(samples[:,0], samples[:,1], p)
    T_all[i,0] = np.median(Tcalc)
    T_all[i,1] = np.percentile(Tcalc, 25)
    T_all[i,2] = np.percentile(Tcalc, 75)

    in_sample['pred_label'] = (in_sample['prob']>p).astype(int)
    cs = pd.crosstab(in_sample['pred_label'],df_obs['sat_bin'])
    sens[i] = cs[1][1]/(cs[1][1] + cs[1][0])
    spec[i] = cs[0][0]/(cs[0][0] + cs[0][1])

    dist[i] = abs(sens[i]-(1-spec[i]))

#%% plot TPR-FPR and transmissivity

s1 = 8
s2 = (5,4)

fig, ax = plt.subplots(figsize=s2)
sc = ax.scatter(1-spec, sens, c=p_all, s=s1, vmin=0.0, vmax=0.9)
ax.axline([0,0], [1,1], color='k', linestyle='--', label='1:1')
ax.set_xlabel('False Positive Ratio (FPR)')
ax.set_ylabel('True Positive Ratio (TPR)')
fig.colorbar(sc, label=r'$p^*$')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig('%s/%s/FPR_TPR_thresh_%d.png'%(directory, base_output_path, ID), transparent=True, dpi=300)
plt.savefig('%s/%s/FPR_TPR_thresh_%d.pdf'%(directory, base_output_path, ID), transparent=True, dpi=300)

fig, ax = plt.subplots(figsize=s2)
sc = ax.scatter(1-spec, sens, c=T_all[:,0], s=s1, cmap='plasma', norm=colors.LogNorm())
ax.axline([0,0], [1,1], color='k', linestyle='--', label='1:1')
ax.set_xlabel('False Positive Ratio (FPR)')
ax.set_ylabel('True Positive Ratio (TPR)')
fig.colorbar(sc, label='Transmissivity')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig('%s/%s/FPR_TPR_trans_%d.png'%(directory, base_output_path, ID), transparent=True, dpi=300)
plt.savefig('%s/%s/FPR_TPR_trans_%d.pdf'%(directory, base_output_path, ID), transparent=True, dpi=300)

fig, ax = plt.subplots(figsize=s2)
sc = ax.scatter(T_all[:,0], dist, c=p_all, s=s1, vmin=0.0, vmax=0.9)
# ax.plot(T_all[:,0], dist, 'k', linewidth=0.5, zorder=100)
ax.set_xscale('log')
ax.set_xlabel('Transmissivity')
ax.set_ylabel('TPR-FPR')
fig.colorbar(sc, label=r'$p^*$')
fig.tight_layout()
plt.savefig('%s/%s/trans_dist_thresh_%d.png'%(directory, base_output_path, ID), transparent=True, dpi=300)
plt.savefig('%s/%s/trans_dist_thresh_%d.pdf'%(directory, base_output_path, ID), transparent=True, dpi=300)

#%% select max transmissivity and save

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
dfT.to_csv(f'{directory}/{base_output_path}/transmissivity_{base_output_path}_logTIQ_{ID}.csv', float_format='%.3f')

# %%
