
#%%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import shutil
from scipy.stats import norm, truncnorm, ranksums
from calc_storm_stats import get_event_interevent_arrays

# make figures?
make_figures = False

# model suffix (CaseStudy_cross_N)
N = 11

# Docs path
path_docs = '/Users/dlitwin/Documents'

# path to save things
figpath = os.path.join(path_docs, 'Papers/Ch3_oregon_ridge_soldiers_delight/figures/')
folder_path = os.path.join(path_docs, 'Research Data/HPC output/DupuitLEMResults/CaseStudy/')

# data paths for DR and BR
path_DR = os.path.join(path_docs, 'Research/Soldiers Delight/data/')
path_BR = os.path.join(path_docs, 'Research/Oregon Ridge/data/')

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

path_ch = os.path.join(path_docs, 'Papers/Ch3_oregon_ridge_soldiers_delight/')
name = 'denudation_piedmont_portenga_2019_fig_4.csv'
df_U = pd.read_csv(path_ch+name) # (Mg km-2 yr-1)
df_U['U'] = df_U[' 10be_rate'] * 1e3 * 1e-6 * (1/2700) * 1/(365*24*3600) # kg/Mg * km2/m2 * m2/kg * yr/sec


# method for calculating std from q1 and q3 https://stats.stackexchange.com/a/256496
n = 10 # number of samples of piedmont in Portenga et al.
U_std = (df_U['U'][4] - df_U['U'][1]) / (2 * norm.ppf((0.75 * n - 0.125) / (n + 0.25)))
U_mean = df_U['U'][2]

df_params['U'] = df_U['U'][2]

# generate truncated normally distributed random samples for erosion/uplift rate https://stackoverflow.com/a/64621360
#U_gen = U_std * np.random.randn(10000) + U_mean
a = (0 - U_mean) / U_std
U_gen = truncnorm.rvs(a, np.inf, loc=U_mean, scale=U_std, size=10000, random_state=1234)


# %% D: Diffusivity

# # rock bulk density:
# rho_serp = 2.6 # average for serpentines
# rho_schist = 2.7 # avg for mica schist g/cm3

# # regolith bulk density
# df_DR_bulk = pd.read_csv(path_DR+'SoilSurvey/BD_depth.csv')
# rho_DR = calc_weighted_avg(df_DR_bulk, 'Rating (grams per\ncubic centimeter)')
# df_BR_bulk = pd.read_csv(path_BR+'SoilSurvey/BD_depth.csv')
# rho_BR = calc_weighted_avg(df_BR_bulk, 'Rating (grams per\ncubic centimeter)')

# # calculate bulk density
# df_params['rho_ratio'] = [rho_serp/rho_DR, rho_schist/rho_BR]

# load hilltop data
path1 = os.path.join(path_DR, 'LSDTT')
path2 = os.path.join(path_BR, 'LSDTT')
name_ht_DR = "baltimore2015_DR1_HilltopData_TN.csv"
name_ht_BR = "baltimore2015_BR_HilltopData_TN.csv"

# pick the right basin and remove some very negative outliers
df_ht_DR = pd.read_csv(os.path.join(path1, name_ht_DR))
df_ht_DR = df_ht_DR[df_ht_DR['BasinID']==99]
df_ht_BR = pd.read_csv(os.path.join(path2, name_ht_BR))
df_ht_BR = df_ht_BR[df_ht_BR['BasinID']==71]
cBR = df_ht_BR['Cht'] > -1
cDR = df_ht_DR['Cht'] > -1

# generate random samples for curvature by selecting with replacement
np.random.seed(123)
Cht_DR_gen = np.random.choice(df_ht_DR['Cht'][cDR], 10000, replace=True)
Cht_BR_gen = np.random.choice(df_ht_BR['Cht'][cBR], 10000, replace=True)

q25_C_DR, med_C_DR, q75_C_DR = np.percentile(df_ht_DR['Cht'][cDR], [25, 50, 75])
q25_C_BR, med_C_BR, q75_C_BR = np.percentile(df_ht_BR['Cht'][cBR], [25, 50, 75])

df_Cht = pd.DataFrame(data=[[q25_C_DR, med_C_DR, q75_C_DR], [q25_C_BR, med_C_BR, q75_C_BR]], 
                    columns=['q25','q50','q75'], index=['DR','BR'])

# calculate diffusivity for each combination
# D_DR = (df_params['rho_ratio']['DR'] * U_gen)/(-Cht_DR_gen)
# D_BR = (df_params['rho_ratio']['BR'] * U_gen)/(-Cht_BR_gen)

# calculate diffusivity for each combination. These are effective diffusivities, because 
# we are not accounting for the difference in bulk density
D_DR = U_gen/(-Cht_DR_gen)
D_BR = U_gen/(-Cht_BR_gen)

# get quantiles
q25_DR, med_DR, q75_DR = np.percentile(D_DR, [25, 50, 75])
q25_BR, med_BR, q75_BR = np.percentile(D_BR, [25, 50, 75])

df_D = pd.DataFrame(data=[[q25_DR, med_DR, q75_DR], [q25_BR, med_BR, q75_BR]], 
                    columns=['q25','q50','q75'], index=['DR','BR'])

# the calculated values look indistinguishable given uncertainty, so let's just
# say that they are the same
df_params['D'] =  df_D['q50'] #df_D['q50'].mean()
D_Stat = ranksums(D_DR, D_BR)

df_params['Sc'] = 1.0

#%% violin plot Cht and D

if make_figures:

    pos   = [1, 2]
    label = ['Druids Run', 'Baisman Run']
    clrs = ['firebrick', 'royalblue']
    Cht = [np.log10(-df_ht_DR['Cht'][cDR]), np.log10(-df_ht_BR['Cht'][cBR])]
    D = [np.log10(D_DR*(3600*24*365)), np.log10(D_BR*(3600*24*365))]

    fig, axs = plt.subplots(ncols=2, figsize=(5,3.5))
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
    dfCht.to_csv(os.path.join(path_ch, 'df_Cht_stats.csv'), float_format="%.3e")
    axs[0].vlines(pos, np.log10(-np.array([DRq1, BRq1])), np.log10(-np.array([DRq3, BRq3])), color='k', linestyle='-', lw=5)
    axs[0].set_ylim((-8,0))
    axs[0].set_xticks(pos)
    axs[0].set_xticklabels(label)
    axs[0].set_ylabel(r'$log_{10} (-C_{ht})$ (1/m)')
    axs[0].set_title('Hilltop Curvature')

    parts = axs[1].violinplot(D, pos, vert=True, showmeans=False, showmedians=True,
            showextrema=True)
    for pc, color in zip(parts['bodies'], clrs):
        pc.set_facecolor(color)
        pc.set_alpha(0.8)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolor("k")
        vp.set_linewidth(1)
    DRq1, med_DR, DRq3 = np.percentile(D_DR*(3600*24*365), [25, 50, 75])
    BRq1, med_BR, BRq3 = np.percentile(D_BR*(3600*24*365), [25, 50, 75])
    axs[1].vlines(pos, np.log10(np.array([DRq1, BRq1])), np.log10(np.array([DRq3, BRq3])), color='k', linestyle='-', lw=5)
    axs[1].set_ylim((-8,3))
    axs[1].set_xticks(pos)
    axs[1].set_xticklabels(label)
    axs[1].set_ylabel(r'$\log_{10}D \,\, (m^2/yr)$')
    axs[1].set_title('Hillslope Diffusivity')
    plt.show()
    fig.tight_layout()
    plt.savefig(figpath+'Cht_D_violinplot.png', dpi=500, transparent=True)
    plt.savefig(figpath+'Cht_D_violinplot.pdf', transparent=True)


# %% K, m, n: streampower coefficients

# import from chi analysis
name_chi_DR = "baltimore2015_DR1_0.5_MChiSegmented.csv"
name_chi_BR = "baltimore2015_BR_0.5_MChiSegmented.csv"
df_chi_DR = pd.read_csv(os.path.join(path1, name_chi_DR))
df_chi_BR = pd.read_csv(os.path.join(path2, name_chi_BR))

# choose steepness index of the segments with larger drainage areas, because 
# this avoids headwaters where there should be more dependence on Q*
Quant_DR = np.quantile(df_chi_DR['drainage_area'], 0.4)
Quant_BR = np.quantile(df_chi_BR['drainage_area'], 0.4)

ksn_DR = df_chi_DR['m_chi'].loc[df_chi_DR['drainage_area']>Quant_DR]
ksn_BR = df_chi_BR['m_chi'].loc[df_chi_BR['drainage_area']>Quant_BR]

np.random.seed(1234)
ksn_DR_gen = np.random.choice(ksn_DR, 10000, replace=True)
ksn_BR_gen = np.random.choice(ksn_BR, 10000, replace=True)

# for DR the best-fit concavity from chi analysis is 0.4, BR did not converge
# use 0.5 because n=/=1 will probably break the nondimensionalization
# can relax this later
df_params['concavity'] = [0.5, 0.5]

df_params['m_sp'] = 0.5 # requirement for DupuitLEM as currently formulated
df_params['n_sp'] = df_params['m_sp']/df_params['concavity'] 
# if you use concavity = 0.4, get n>1, consistent with Harel, Mudd, Attal 2016

# calculate total erodibility
Ksp_DR = U_gen/ksn_DR_gen**df_params['n_sp']['DR'] # from streampower law, this is the total erodibility coefficient
Ksp_BR = U_gen/ksn_BR_gen**df_params['n_sp']['BR']

# get quantiles and make dataframe
q25_DR, med_DR, q75_DR = np.percentile(Ksp_DR, [25, 50, 75])
q25_BR, med_BR, q75_BR = np.percentile(Ksp_BR, [25, 50, 75])
df_Ksp = pd.DataFrame(data=[[q25_DR, med_DR, q75_DR], [q25_BR, med_BR, q75_BR]], 
                    columns=['q25','q50','q75'], index=['DR','BR'])

# calculate statistic
Ksp_Stat = ranksums(Ksp_DR, Ksp_BR)

#%% model parameters from chi analysis

# estimate the runoff ratio Qstar_max = (Q/P) for the sites, which is needed to isolate K from Q* in our streampower law
# start by assuming that they are the same, then iterate based on the model results.
df_params['Qstar_max'] = [0.6,0.3] #0.3,0.3


# Lh for hillslope length correction
lh_path = '/Users/dlitwin/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/df_Lh_stats.csv'
df_Lh_stats = pd.read_csv(lh_path, index_col=0)

r_path = '/Users/dlitwin/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/df_Relief_stats.csv'
df_R_stats = pd.read_csv(r_path, index_col=0)

# calculate erodibility based on Qstar_max
df_params['v0'] = 10 # 30 15 5 # window we averaged DEMs over to calculate most quantities
df_params['Ksp'] = df_Ksp['q50']
df_params['K'] = df_params['Ksp']/df_params['Qstar_max'] # the coefficient we use has to be greater because it will be multiplied by Q*
# df_params['K'] = (df_params['Ksp']*(1+df_Lh_stats['q50']/df_params['v0']))/df_params['Qstar_max'] # scale with Q*, but also scale by hillslope length factor


#%% Violin plot - total erodibility

if make_figures:
        
    pos   = [1, 2]
    label = ['Druids Run', 'Baisman Run']
    clrs = ['firebrick', 'royalblue']

    K = [np.log10(Ksp_DR*(3600*24*365)), np.log10(Ksp_BR*(3600*24*365))]
    q25_DR, med_DR, q75_DR = np.percentile(Ksp_DR, [25, 50, 75])
    q25_BR, med_BR, q75_BR = np.percentile(Ksp_BR, [25, 50, 75])

    fig, ax = plt.subplots(figsize=(2.5,3.5))
    parts = ax.violinplot(K, pos, vert=True, showmeans=False, showmedians=True,
            showextrema=True)
    for pc, color in zip(parts['bodies'], clrs):
        pc.set_facecolor(color)
        pc.set_alpha(0.8)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolor("k")
        vp.set_linewidth(1)
    ax.vlines(pos, np.log10(np.array([q25_DR, q25_BR])*(3600*24*365)), 
            np.log10(np.array([q75_DR, q75_BR])*(3600*24*365)), color='k', linestyle='-', lw=5)
    ax.set_ylim((-10,-2))
    ax.set_xticks(pos)
    ax.set_xticklabels(label)
    ax.set_ylabel(r'$\log_{10}(K_{sp} \,\, (1/yr))$')
    ax.set_title('Total Erodibility')

    plt.show()
    fig.tight_layout()
    plt.savefig(figpath + 'Ksp_violinplot.png', dpi=500)


#%% Violin plot - steepness ksn and erodibility K

if make_figures:
    pos   = [1, 2]
    label = ['Druids Run', 'Baisman Run']
    clrs = ['firebrick', 'royalblue']

    Qm = df_params['Qstar_max'].values #[0.6,0.3] # 
    K = [np.log10(Ksp_DR*(3600*24*365)/Qm[0]), np.log10(Ksp_BR*(3600*24*365)/Qm[1])]
    ksn = [np.log10(ksn_DR), np.log10(ksn_BR)]
    # ksn = [ksn_DR, ksn_BR]


    fig, axs = plt.subplots(ncols=2, figsize=(6,3.5))
    parts = axs[0].violinplot(ksn, pos, vert=True, showmeans=False, showmedians=True,
            showextrema=True)
    for pc, color in zip(parts['bodies'], clrs):
        pc.set_facecolor(color)
        pc.set_alpha(0.8)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolor("k")
        vp.set_linewidth(1)
    DRq1, DRmed, DRq3 = np.percentile(ksn_DR, [25, 50, 75])
    BRq1, BRmed, BRq3 = np.percentile(ksn_BR, [25, 50, 75])
    dfksn = pd.DataFrame(data=[[DRq1, DRmed, DRq3, ksn_DR.mean()], [BRq1, BRmed, BRq3, ksn_BR.mean()]], 
                        columns=['q25','q50','q75', 'mean'], index=['DR','BR'])
    dfksn.to_csv(os.path.join(path_ch+'df_ksn_stats.csv'))
    axs[0].vlines(pos, np.log10(np.array([DRq1, BRq1])), np.log10(np.array([DRq3, BRq3])), color='k', linestyle='-', lw=5)
    # axs[0].vlines(pos, np.array([DRq1, BRq1]), np.array([DRq3, BRq3]), color='k', linestyle='-', lw=5)
    # axs[0].set_ylim((-8,0))
    axs[0].set_xticks(pos)
    axs[0].set_xticklabels(label)
    axs[0].set_ylabel(r'$log_{10} (k_{sn})$ (-)')
    axs[0].set_title('Steepness index')

    parts = axs[1].violinplot(K, pos, vert=True, showmeans=False, showmedians=True,
            showextrema=True)
    for pc, color in zip(parts['bodies'], clrs):
        pc.set_facecolor(color)
        pc.set_alpha(0.8)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolor("k")
        vp.set_linewidth(1)
    DRq1, DRmed, DRq3 = np.percentile(Ksp_DR*(3600*24*365)/Qm[0], [25, 50, 75])
    BRq1, BRmed, BRq3 = np.percentile(Ksp_BR*(3600*24*365)/Qm[1], [25, 50, 75])
    dfK = pd.DataFrame(data=[[DRq1, DRmed, DRq3], [BRq1, BRmed, BRq3]], 
                        columns=['q25','q50','q75'], index=['DR','BR'])
    dfK.to_csv(os.path.join(path_ch+'df_K_stats.csv'))
    axs[1].vlines(pos, np.log10(np.array([DRq1, BRq1])), np.log10(np.array([DRq3, BRq3])), color='k', linestyle='-', lw=5)
    axs[1].set_ylim((-9.5,-3))
    axs[1].set_xticks(pos)
    axs[1].set_xticklabels(label)
    axs[1].set_ylabel(r'$\log_{10}(K)$ (1/yr)')
    axs[1].set_title('Incision coefficient')
    plt.show()
    fig.tight_layout()
    plt.savefig(figpath + 'ksn_K_violinplot.png', dpi=300, transparent=True)
    plt.savefig(figpath + 'ksn_K_violinplot.pdf', transparent=True)

# %% precipitation

path = os.path.join(path_docs, "Research/Transmissivity wetness/Data/BAIS/BAIS_PQ_hr.p")
df_P = pickle.load(open(path,"rb"))

storm_depths, storm_durs, interstorm_durs = get_event_interevent_arrays(df_P, 'Precip mm/hr')

if make_figures:
    fig, axs = plt.subplots(ncols=3)
    axs[0].hist(storm_depths, bins=25, density=True)
    axs[0].set_xlabel('storm depth (mm)')
    axs[1].hist(storm_durs, bins=25, density=True)
    axs[1].set_xlabel('storm duration (hr)')
    axs[2].hist(interstorm_durs, bins=25, density=True)
    axs[2].set_xlabel('interstorm duration (hr)')
    plt.show()

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


# %% permeable thickness and hydraulic conductivity

# for DR, select the dominant soil type, chrome silt loam. All slope classes (CeB, CeC, CeD)
# have the same values, so aggregate them together. There's a permeability contrast at the A
# horizon so we'll focus on flow above that layer
df_DR_soil = pd.read_csv(path_DR+'SoilSurvey/SoilHydraulicProperties_DR.csv')
df_DR_soil = df_DR_soil[df_DR_soil['musym'].isin(['CeB', 'CeC', 'CeD'])]
df_grouped = df_DR_soil.groupby('desgnmaster').mean(numeric_only=True)
# b_DR = df_grouped.loc['A']['hzdepb_r'] * 0.0254 #m
b_DR = 0.2 #m make this very small 

# transmissivity we calculated using the saturation survey data
dfT_DR = pd.read_csv(figpath+'transmissivity_DR_logTIQ_5.csv')
ksat_DR = (dfT_DR['Trans. med [m2/d]'][0]/(24*3600))/b_DR  #m/s

# for Baisman Run, soil survey data is too shallow, so we will have to do
# some of our own estimates. Here I'm assuming soil goes to the maximum depth reported.
# this is likely not quite true - would be better to talk to Cassie or get the actual
# horizon depths from the survey
df_BR_depth = pd.read_csv(path_BR+'SoilSurvey/Bedrock_depth.csv')
b_soil = calc_weighted_avg(df_BR_depth, 'Rating (centimeters) ') / 100 # meters
b_saprolite = 2 # meters. An estimate for average thickness
b_BR = b_soil + b_saprolite

# transmissivity we calculated using the saturation survey data
dfT_BR = pd.read_csv(figpath+'transmissivity_BR_logTIQ_5.csv')
ksat_BR = (dfT_BR['Trans. med [m2/d]'][0]/(24*3600))/b_BR  #m/s

# test a case where transmissivity at BR half of what we found
# ksat_BR = (0.5*dfT_BR['Trans. med [m2/d]'][0]/(24*3600))/b_BR  #m/s

df_params['ksat'] = [ksat_DR, ksat_BR]
df_params['b'] = [b_DR, b_BR]

# %% water contents: ne and na

# water capacity we take directly from the depth averaged values from the soil survey.
# they are not as different between the two sites as I Thought they might be. This is a 
# parameter that could be calibrated to get water balance right.
df_DR_na = pd.read_csv(path_DR+'SoilSurvey/Capacity_avg.csv')
# na_DR = calc_weighted_avg(df_DR_na, 'Rating (centimeters per\ncentimeter)')
na_DR = 0.19 # AWC for A horizon

df_BR_na = pd.read_csv(path_BR+'SoilSurvey/Capacity_avg.csv')
na_BR = calc_weighted_avg(df_BR_na, 'Rating (centimeters per\ncentimeter)')

df_params['na'] = [na_DR, na_BR]

# the drainable porosity is estimated by first coming up with the total porosity using
# the bulk density and an assumed soil particle density. Then we subtract the available
# water content, which will stay under tension when the water table falls.
particle_density = 2.65 #g/cm3 average particle density (as in https://www.usgs.gov/data/soil-properties-dataset-united-states)
df_DR_bd = pd.read_csv(path_DR+'SoilSurvey/BD_surface.csv')
# bd_DR = calc_weighted_avg(df_DR_bd, 'Rating (grams per\ncubic centimeter)')
bd_DR = 1.15 # representative for A horizon
ne_DR = (1 - bd_DR/particle_density) - na_DR


df_BR_bd = pd.read_csv(path_BR+'SoilSurvey/BD_surface.csv')
bd_BR = calc_weighted_avg(df_BR_bd, 'Rating (grams per\ncubic centimeter)')
ne_BR = (1 - bd_BR/particle_density) - na_BR

# df_params['ne'] = [ne_DR, ne_BR]
df_params['ne'] = [0.25, 0.25]

#%% potential evapotranspiration

# https://www.nrcc.cornell.edu/wxstation/pet/pet.html
pet_month = np.array([0.59,0.83,1.68,2.69,3.96,4.43,4.76,4.11,2.90,1.88,0.96,0.58]) * 0.0254 # m
# this is consistent with older data https://semspub.epa.gov/work/01/554363.pdf

# average and divide by the fraction of time that there is pet (interstorm periods)
df_params['pet'] = np.sum(pet_month)/(365*24*3600) / (df_params['tb']/(df_params['tr']+df_params['tb'])) # avg m/s


#%% add swapped model runs

"""
The matrix here is:
                    hydro
               |__DR___|___BR___   
geomorph    DR | 'DR'  |  'BRx'
            BR | 'DRx' |  'BR'

where geomorphic is changing K and D, since we assume U and v0 are the same
"""

df_params.loc['DRx'] = df_params.loc['DR']
df_params['K'].loc['DRx'] = df_params['K'].loc['BR']
df_params['D'].loc['DRx'] = df_params['D'].loc['BR']

df_params.loc['BRx'] = df_params.loc['BR']
df_params['K'].loc['BRx'] = df_params['K'].loc['DR']
df_params['D'].loc['BRx'] = df_params['D'].loc['DR']

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

dtg_max_nd = 1e-3 # maximum geomorphic timestep in units of tg [-]
Th_nd = 25 # hydrologic time in units of (tr+tb) [-]
bin_capacity_nd = 0.05 # bin capacity as a proportion of mean storm depth

df_params['Nx'] = 200 #300 # number of grid cells width and height
df_params['Nz'] = round((df_params['b']*df_params['na'])/(bin_capacity_nd*df_params['ds']))
df_params['Tg'] = 5e7*(365*24*3600) # 5e7 Tg_nd*df_params['tg'] # Total geomorphic simulation time [s] #5e6
df_params['ksf'] = 10000 #morphologic scaling factor
df_params['Th'] = Th_nd*(df_params['tr']+df_params['tb']) # hydrologic simulation time [s]
df_params['dtg'] = df_params['ksf']*df_params['Th'] # geomorphic timestep [s]
df_params['dtg_max'] = dtg_max_nd*df_params['tg'] # the maximum duration of a geomorphic substep [s]
df_params['output_interval'] = 2000 #(10/(df_params['dtg']/df_params['tg'])).round().astype(int)
df_params['BCs'] = '4414' # set boundary conditions

# %% set index

df_params['label'] = df_params.index.values
df_params.set_index(np.arange(len(df_params)), inplace=True)
df_params.drop(columns=['label'], inplace=True)

#%% create folders and save parameter files

for i in df_params.index:

    name = 'CaseStudy_cross_%d-%d'%(N,i)
    outpath = folder_path+name
    # outpath = os.path.join(folder_path, os.sep, name)
    if not os.path.exists(outpath):
        os.mkdir(outpath) 
    df_params.loc[i].to_csv(outpath+'/parameters.csv', index=True)

    shutil.copyfile(__file__, outpath+'/ch3_model_parameters.py')
# %%
