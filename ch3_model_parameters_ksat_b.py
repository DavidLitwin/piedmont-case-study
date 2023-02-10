
#%%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from itertools import product

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

#%% ksat and b

ksat_range = np.geomspace(0.3, 1.2, 5) * 1/(24*3600)
b_range = np.geomspace(1.5, 6, 5)

# ksat_range = np.geomspace(2, 10, 5) * 1/(24*3600)
# b_range = np.geomspace(0.25, 1.25, 5)

ksat_all = []
b_all = []
for ksat, b in product(ksat_range,b_range):
    ksat_all.append(ksat)
    b_all.append(b)


df_params = pd.DataFrame()
df_params['ksat'] = ksat_all
df_params['b'] = b_all
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
rho_ratio = np.array([rho_serp/rho_DR, rho_schist/rho_BR])

# use the median of the hilltop curvature - not that different from log-mean
df_HT_stats = pd.read_csv(path+"df_HT_stats.csv", index_col=0)
D = (rho_ratio*df_params['U'][0])/(-df_HT_stats['Med'])

# the calculated values are indistinguishable given uncertainty, so let's just
# say that they are the same
df_params['D'] = np.mean(D)
# %% K, m, n: streampower coefficients

path1 = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/LSDTT/'
path2 = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/LSDTT/'

name_chi_DR = "baltimore2015_DR1_0.5_MChiSegmented.csv"
name_chi_BR = "baltimore2015_BR_0.5_MChiSegmented.csv"

df_chi_DR = pd.read_csv(path1 + name_chi_DR)
df_chi_BR = pd.read_csv(path2 + name_chi_BR)


#%%

# estimate the runoff ratio Qstar_max = (Q/P) for the sites, which is needed to isolate K from Q* in our streampower law
# start by assuming that they are the same, then iterate based on the model results.
df_params['Qstar_max'] = 0.3

# choose steepness index of the segments with larger drainage areas, because 
# this avoids headwaters where there should be more dependence on Q*
Quant_BR = np.quantile(df_chi_BR['drainage_area'], 0.8)
Quant_DR = np.quantile(df_chi_DR['drainage_area'], 0.8)

# use median to decrease the effect of some anomalous reaches associated with
# lithological difference or possibly transience
ksn_BR = df_chi_BR['m_chi'].loc[df_chi_BR['drainage_area']<Quant_BR].median()
ksn_DR = df_chi_DR['m_chi'].loc[df_chi_DR['drainage_area']>Quant_DR].median()
df_params['ksn'] = ksn_BR

# for DR the best-fit concavity from chi analysis is 0.4, BR did not converge
# use 0.5 because n=/=1 will probably break the nondimensionalization
# can relax this later
df_params['concavity'] = 0.5

df_params['m_sp'] = 0.5 # requirement for DupuitLEM as currently formulated
df_params['n_sp'] = df_params['m_sp']/df_params['concavity'] 
# if you use concavity = 0.4, get n>1, consistent with Harel, Mudd, Attal 2016

Ksp = df_params['U']/df_params['ksn']**df_params['n_sp'] # from streampower law, this is the total erodibility coefficient
df_params['K'] = Ksp/df_params['Qstar_max'] # the coefficient we use has to be greater because it will be multiplied by Q*
df_params['v0'] = 10 # window we averaged DEMs over to calculate most quantities


# %% precipitation

path = "C:/Users/dgbli/Documents/Research/Transmissivity wetness/Data/BAIS/BAIS_PQ_hr.p"
df_P = pickle.load(open(path,"rb"))

storm_depths, storm_durs, interstorm_durs = get_event_interevent_arrays(df_P, 'Precip mm/hr')

# simplest possible way of calculating event statistics. In reality, more work
# could be put into fitting an exponential model
df_params['tr'] = np.mean(storm_durs)*3600
df_params['tb'] = np.mean(interstorm_durs)*3600
df_params['ds'] = np.mean(storm_depths)*0.001
df_params['p'] = df_params['ds']/(df_params['tr'] + df_params['tb']) 
# df_params['p'] = np.mean(df_P['Precip mm/hr'])*0.001*(1/3600) # this should be close to the estimate above
# check if these are consistent with Leclerc (1973) thesis.

# from USGS streamstats
# df_params['p'] = 45.6 * 0.0254 / (3600*24*365) # m/s


#%% potential evapotranspiration

# https://www.nrcc.cornell.edu/wxstation/pet/pet.html
pet_month = np.array([0.59,0.83,1.68,2.69,3.96,4.43,4.76,4.11,2.90,1.88,0.96,0.58]) * 0.0254 # m
# this is consistent with older data https://semspub.epa.gov/work/01/554363.pdf

# average and divide by the fraction of time that there is pet (interstorm periods)
df_params['pet'] = np.sum(pet_month)/(365*24*3600) / (df_params['tb']/(df_params['tr']+df_params['tb'])) # avg m/s

# %% water contents: ne and na

# water capacity we take directly from the depth averaged values from the soil survey.
# they are not as different between the two sites as I Thought they might be. This is a 
# parameter that could be calibrated to get water balance right.
df_DR_na = pd.read_csv(path_DR+'SoilSurvey/Capacity_avg.csv')
na_DR = calc_weighted_avg(df_DR_na, 'Rating (centimeters per\ncentimeter)')

df_BR_na = pd.read_csv(path_BR+'SoilSurvey/Capacity_avg.csv')
na_BR = calc_weighted_avg(df_BR_na, 'Rating (centimeters per\ncentimeter)')

df_params['na'] = np.mean([na_BR,na_DR])

# the drainable porosity is estimated by first coming up with the total porosity using
# the bulk density and an assumed soil particle density. Then we subtract the available
# water content, which will stay under tension when the water table falls.
particle_density = 2.65 #g/cm3 average particle density (as in https://www.usgs.gov/data/soil-properties-dataset-united-states)
df_DR_bd = pd.read_csv(path_DR+'SoilSurvey/BD_surface.csv')
bd_DR = calc_weighted_avg(df_DR_bd, 'Rating (grams per\ncubic centimeter)')
ne_DR = (1 - bd_DR/particle_density) - na_DR

df_BR_bd = pd.read_csv(path_BR+'SoilSurvey/BD_surface.csv')
bd_BR = calc_weighted_avg(df_BR_bd, 'Rating (grams per\ncubic centimeter)')
ne_BR = (1 - bd_BR/particle_density) - na_BR

df_params['ne'] = 0.2 #np.mean([ne_DR, ne_BR])

# %% characteristic scales + dimless groups

df_params['lg'] = (df_params['D']**2/(df_params['v0']*df_params['K']**2))**(1/3)
df_params['hg'] = ((df_params['D']*df_params['U']**3)/(df_params['v0']**2*df_params['K']**4))**(1/3)
df_params['tg'] = ((df_params['D'])/(df_params['v0']**2*df_params['K']**4))**(1/3)
df_params['ha'] = (df_params['p']*df_params['lg'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer thickness [m]
df_params['td'] = (df_params['lg']*df_params['ne'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer drainage time [s]

df_params['alpha'] = df_params['hg']/df_params['lg']
df_params['gam'] = (df_params['b']*df_params['ksat']*df_params['hg'])/(df_params['p']*df_params['lg']**2)
df_params['beta'] = (df_params['ksat']*df_params['hg']**2)/(df_params['p']*df_params['lg']**2)
df_params['sigma'] = (df_params['b']*df_params['ne'])/(df_params['ds'])
df_params['rho'] = df_params['tr']/(df_params['tr']+df_params['tb'])
df_params['ai'] = (df_params['pet']/df_params['p']) * df_params['tb']/(df_params['tr']+df_params['tb'])
df_params['phi'] = df_params['na']/df_params['ne']

#%% Operational parameters 

dtg_max_nd = 2e-3 # maximum geomorphic timestep in units of tg [-]
Th_nd = 25 # hydrologic time in units of (tr+tb) [-]
bin_capacity_nd = 0.05 # bin capacity as a proportion of mean storm depth

df_params['Nx'] = 125 # number of grid cells width and height
df_params['Nz'] = round((df_params['b']*df_params['na'])/(bin_capacity_nd*df_params['ds']))
df_params['Tg'] = 5e7*(365*24*3600) # Tg_nd*df_params['tg'] # Total geomorphic simulation time [s]
df_params['ksf'] = 5000 # morphologic scaling factor
df_params['Th'] = Th_nd*(df_params['tr']+df_params['tb']) # hydrologic simulation time [s]
df_params['dtg'] = df_params['ksf']*df_params['Th'] # geomorphic timestep [s]
df_params['dtg_max'] = dtg_max_nd*df_params['tg'] # the maximum duration of a geomorphic substep [s]
df_params['output_interval'] = (10/(df_params['dtg']/df_params['tg'])).round().astype(int)

# %%

# df_params['label'] = df_params.index.values
# df_params.set_index(np.arange(len(df_params)), inplace=True)
# df_params.drop(columns=['label'], inplace=True)

#%%

folder_path = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/CaseStudy/'
N = 10


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
