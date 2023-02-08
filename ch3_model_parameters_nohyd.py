
#%%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from calc_storm_stats import get_event_interevent_arrays


path_DR = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/'
path_BR = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/'

def calc_weighted_avg(df, key):
    """calculate the weighted mean of a quantity from a Soil Survey dataframe.
    Assumes that anything listed as '>' is equal to that value."""

    df['prop'] = 0.01 * df['Percent of AOI'].str.strip('%').astype(float)
    
    try:
        weighted_mean = np.sum(df[key]*df['prop'])
    except:
        df[key] = df[key].str.strip('>').astype(float)
        weighted_mean = np.sum(df[key]*df['prop'])

    return weighted_mean

#%%
df_params = pd.DataFrame(index=['DR', 'BR'])
#%% U: Uplift/Baselevel change rate

path = 'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/'
name = 'denudation_piedmont_portenga_2019_fig_4.csv'
df_U = pd.read_csv(path+name) # (Mg km-2 yr-1)
df_U['U'] = df_U[' 10be_rate'] * 1e3 * 1e-6 * (1/2700) * 1/(365*24*3600) # kg/Mg * km2/m2 * m2/kg * yr/sec

df_params['U'] = df_U['U'][2]

# %% D: Diffusivity

# rock bulk density:
rho_serp = 2.6 # average for serpentines
rho_schist = 2.7 # avg for mica schist g/cm3

# regolith bulk density
df_DR_bulk = pd.read_csv(path_DR+'SoilSurvey/BD_depth.csv')
rho_DR = calc_weighted_avg(df_DR_bulk, 'Rating (grams per\ncubic centimeter)')
df_BR_bulk = pd.read_csv(path_BR+'SoilSurvey/BD_depth.csv')
rho_BR = calc_weighted_avg(df_BR_bulk, 'Rating (grams per\ncubic centimeter)')

# calculate bulk density
df_params['rho_ratio'] = [rho_serp/rho_DR, rho_schist/rho_BR]

# use the median of the hilltop curvature - not that different from log-mean
df_HT_stats = pd.read_csv(path+"df_HT_stats.csv", index_col=0)
df_params['D'] = (df_params['rho_ratio']*df_params['U'])/(-df_HT_stats['Med'])

# the calculated values are indistinguishable given uncertainty, so let's just
# say that they are the same
df_params['D'] = df_params['D'].mean()
# %% K, m, n: streampower coefficients

path1 = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/LSDTT/'
path2 = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/LSDTT/'

name_chi_DR = "baltimore2015_DR1_0.4_MChiSegmented.csv"
name_chi_BR = "baltimore2015_BR_0.4_MChiSegmented.csv"

df_chi_DR = pd.read_csv(path1 + name_chi_DR)
df_chi_BR = pd.read_csv(path2 + name_chi_BR)


#%%

# choose steepness index of the segments with larger drainage areas, because 
# this avoids headwaters where there should be more dependence on Q*
Quant_BR = np.quantile(df_chi_BR['drainage_area'], 0.8)
Quant_DR = np.quantile(df_chi_DR['drainage_area'], 0.8)

# use median to decrease the effect of some anomalous reaches associated with
# lithological difference or possibly transience
ksn_BR = df_chi_BR['m_chi'].loc[df_chi_BR['drainage_area']<Quant_BR].median()
ksn_DR = df_chi_DR['m_chi'].loc[df_chi_DR['drainage_area']>Quant_DR].median()
df_params['ksn'] = [ksn_DR, ksn_BR]

# for DR the best-fit concavity from chi analysis is 0.4, BR did not converge
# use 0.5 because n=/=1 will probably break the nondimensionalization
# can relax this later
df_params['concavity'] = [0.4, 0.4]

df_params['m_sp'] = 0.5 # requirement for DupuitLEM as currently formulated
df_params['n_sp'] = df_params['m_sp']/df_params['concavity'] 
# if you use concavity = 0.4, get n>1, consistent with Harel, Mudd, Attal 2016

df_params['K'] = df_params['U']/df_params['ksn']**df_params['n_sp'] # from streampower law, this is the total erodibility coefficient
df_params['v0'] = 10 # window we averaged DEMs over to calculate most quantities

# plt.figure()
# df_chi_BR['m_chi'].plot.density(color='r', label='Baisman')
# df_chi_DR['m_chi'].plot.density(color='b', label='Druids')
# plt.axvline(df_chi_BR['m_chi'].median(), color='r')
# plt.axvline(df_chi_DR['m_chi'].median(), color='b')
# plt.axvline(ksn_BR, color='r', linestyle='--')
# plt.axvline(ksn_DR, color='b', linestyle='--')
# plt.xlabel('$k_{sn}$')
# plt.legend(frameon=False)


# %% characteristic scales + dimless groups

df_params['lg'] = (df_params['D']**2/(df_params['v0']*df_params['K']**2))**(1/3)
df_params['hg'] = ((df_params['D']*df_params['U']**3)/(df_params['v0']**2*df_params['K']**4))**(1/3)
df_params['tg'] = ((df_params['D'])/(df_params['v0']**2*df_params['K']**4))**(1/3)

#%% Operational parameters 

df_params['Tg'] = 5e6*(365*24*3600) # Total geomorphic simulation time [s]
df_params['dtg'] = 50
df_params['output_interval'] = 1000

# %%

df_params['label'] = df_params.index.values
df_params.set_index(np.arange(len(df_params)), inplace=True)
df_params.drop(columns=['label'], inplace=True)

#%%

folder_path = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/CaseStudy/'
N = 6


for i in df_params.index:

    name = 'CaseStudy_%d-%d'%(N,i)
    outpath = folder_path+name
    # outpath = os.path.join(folder_path, os.sep, name)
    if not os.path.exists(outpath):
        os.mkdir(outpath) 
    df_params.loc[i].to_csv(outpath+'/parameters.csv', index=True)



# %% check parameters set 

ID = 0
task_id = '0'

name = 'CaseStudy_%d-%d'%(N,ID)
outpath = folder_path+name

df_params1 = pd.read_csv(outpath+'/parameters.csv', index_col=0)[task_id]


# %%
