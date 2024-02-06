"""
Plots of pebble counts at Druids Run and Baisman Run
"""

#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

directory_SD = '/Users/dlitwin/Documents/Research/Soldiers Delight/data/pebble_counts'
directory_OR = '/Users/dlitwin/Documents/Research/Oregon Ridge/data/pebble_counts'
directory_figs = '/Users/dlitwin/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures'
date = '20231231'

# approximate locations of each survey
locs = [['OR1', 39.4795, -76.6784],
        ['OR2', 39.4784, -76.6899],
        ['SD1', 39.4175, -76.8518],
        ['SD2', 39.4187, -76.8495]]
#%%

df_dict = {}
for site in ['DR', 'BR']:
    for id in ['1', '2']:
        name = f'{site}_pebble_count_{id}_{date}.csv'
        if site=='OR':
            df = pd.read_csv(os.path.join(directory_OR, name))
        else:
            df = pd.read_csv(os.path.join(directory_SD, name))
        df_dict[f'{site}{id}'] = df


# %%

# df = df_dict['SD2']
b = [0,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2]

plt.figure()
for key, df in df_dict.items():
    plt.hist(df['b-axis (cm)'],  bins=b, histtype='step', stacked=True, fill=False, label=key)
plt.xscale('log')
plt.legend(frameon=False)
# %%

clrs = {'SD': 'firebrick', 'OR':'royalblue'}
lstyles = {'1':'-', '2':'-.'}
plt.figure(figsize=(5,4))
for key, df in df_dict.items():
    vals = np.sort(df['b-axis (cm)'])
    index = (np.arange(len(vals))+1)/len(vals)
    plt.plot(vals, index, label=key, color=clrs[key[0:2]], linestyle=lstyles[key[-1]])
plt.legend(frameon=False)
plt.xlabel('Grain b-axis (cm)')
plt.ylabel('Exceedance (-)')
plt.xscale('log')
plt.ylim((0,1))
plt.savefig(os.path.join(directory_figs, 'pebble_counts_cdf.png'), dpi=300, transparent=True)

# %%
