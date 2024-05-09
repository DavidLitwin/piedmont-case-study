#%%

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
from matplotlib import colors

save_directory = '/Users/dlitwin/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/'

#%% get topographic index

site = 'BR'
res = 5

if site=='DR':
    path = '/Users/dlitwin/Documents/Research/Soldiers Delight/data/'
    processed_path = '/Users/dlitwin/Documents/Research/Soldiers Delight/data_processed/'
elif site=='BR':
    path = '/Users/dlitwin/Documents/Research/Oregon Ridge/data/'
    processed_path = '/Users/dlitwin/Documents/Research/Oregon Ridge/data_processed/'

dfnew = pd.read_csv(processed_path + f"saturation_{site}_{res}.csv", index_col=0)

#%% Logistic regression: sat_bin ~ logTIQ

# logTIQ = TI + log(Q) because TI is already log
dfnew['logTIQ'] = dfnew['TI_filtered'] + np.log(dfnew['Q m/d'])

# logit using TI and Q as predictors
model = smf.logit('sat_bin ~ logTIQ', data=dfnew).fit()

# check model performance
print(model.summary())
with open(save_directory+f'summary_{site}_logTIQ_{res}.txt', 'w') as fh:
    fh.write(model.summary().as_text())

# predict in sample
in_sample = pd.DataFrame({'prob':model.predict()})

#%% calculate transmissivity: range of thresholds

N = 1000
if site=='DR':
    p_all = np.linspace(0.2,0.7,N) #DR
else:
    p_all = np.linspace(0.05,0.7,N) #BR

rhostar = lambda p: np.log(p/(1-p))
Tmean = lambda b0, b1, p: np.exp((rhostar(p)-b0)/b1)

T_all = np.zeros((N,3))
sens = np.zeros(N)
spec = np.zeros(N)
dist = np.zeros(N)

covs = model.cov_params()
means = model.params

samples = np.random.multivariate_normal(means, covs, size=100000)

for i, p in enumerate(p_all):

    Tcalc = Tmean(samples[:,0], samples[:,1], p)
    T_all[i,0] = np.median(Tcalc)
    T_all[i,1] = np.percentile(Tcalc, 25)
    T_all[i,2] = np.percentile(Tcalc, 75)

    in_sample['pred_label'] = (in_sample['prob']>p).astype(int)
    cs = pd.crosstab(in_sample['pred_label'],dfnew['sat_bin'])
    sens[i] = cs[1][1]/(cs[1][1] + cs[1][0])
    spec[i] = cs[0][0]/(cs[0][0] + cs[0][1])

    dist[i] = abs(sens[i]-(1-spec[i]))

#%%

s1 = 8
s2 = (5,4)

fig, ax = plt.subplots(figsize=s2)
sc = ax.scatter(1-spec, sens, c=p_all, s=s1, vmin=0.0, vmax=0.7)
ax.axline([0,0], [1,1], color='k', linestyle='--', label='1:1')
ax.set_xlabel('False Positive Ratio (FPR)')
ax.set_ylabel('True Positive Ratio (TPR)')
fig.colorbar(sc, label=r'$p^*$')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig(save_directory+'FPR_TPR_thresh_%s.png'%site, dpi=300, transparent=True)
plt.savefig(save_directory+'FPR_TPR_thresh_%s.pdf'%site, transparent=True)

fig, ax = plt.subplots(figsize=s2)
sc = ax.scatter(1-spec, sens, c=T_all[:,0], s=s1, cmap='plasma', norm=colors.LogNorm())
ax.axline([0,0], [1,1], color='k', linestyle='--', label='1:1')
ax.set_xlabel('False Positive Ratio (FPR)')
ax.set_ylabel('True Positive Ratio (TPR)')
fig.colorbar(sc, label='Transmissivity')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig(save_directory+'FPR_TPR_trans_%s.png'%site, dpi=300, transparent=True)
plt.savefig(save_directory+'FPR_TPR_trans_%s.pdf'%site, transparent=True)

fig, ax = plt.subplots(figsize=s2)
sc = ax.scatter(T_all[:,0], dist, c=p_all, s=s1, vmin=0.0, vmax=0.7)
# ax.plot(T_all[:,0], dist, 'k', linewidth=0.5, zorder=100)
ax.set_xscale('log')
ax.set_xlabel('Transmissivity')
ax.set_ylabel('TPR-FPR')
fig.colorbar(sc, label=r'$p^*$')
fig.tight_layout()
plt.savefig(save_directory+'trans_dist_thresh_%s.png'%site, dpi=300, transparent=True)
plt.savefig(save_directory+'trans_dist_thresh_%s.pdf'%site, transparent=True)

#%% select max and save

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
dfT.to_csv(save_directory+f'transmissivity_{site}_logTIQ_{res}.csv', float_format='%.3f')


# %%
