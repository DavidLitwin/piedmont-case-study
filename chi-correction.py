"""
Check DupuitLEM results for a correction to Chi that accounts for hillslopes

"""
#%%
import os 
import glob
import numpy as np
import pandas as pd

from matplotlib import cm, colors, ticker
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)

# directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
directory = '/Users/dlitwin/Documents/Research Data/HPC output/DupuitLEMResults/post_proc/'
base_output_path = 'steady_sp_3_18' #'CaseStudy_cross_6'
model_runs = np.arange(30).reshape((5,6))[0::2,2:].flatten()

names = ['DR-DR', 'BR-BR', 'DR-BR', 'BR-DR'] # in order

#%% Load params and output

# results
dfs = []
for ID in model_runs:
    try:
        df = pd.read_csv('%s/%s/output_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
    except FileNotFoundError:
        df =  pd.DataFrame(columns=df.columns)
    dfs.append(df)
df_results = pd.concat(dfs, axis=1, ignore_index=True).T

# params
# dfs = []
# for ID in model_runs:
#     df = pd.read_csv('%s/%s/params_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
#     dfs.append(df)
# df_params = pd.concat(dfs, axis=1, ignore_index=True).T
df_params = pd.read_csv('%s/%s/parameters.csv'%(directory,base_output_path), index_col=0)

# hilltops
files_ht = ["%s-%d_pad_HilltopData_TN.csv"%(base_output_path, i) for i in model_runs]
df_ht_all = [pd.read_csv(os.path.join(directory,base_output_path,file_ht)) for file_ht in files_ht]

# channels
files_chi = ["%s-%d_pad_MChiSegmented.csv"%(base_output_path, i) for i in model_runs]
df_chi_all = [pd.read_csv(os.path.join(directory,base_output_path,file_chi)) for file_chi in files_chi]

#%% calculate apparent K and Ksp

Ksp_all = []
K_all = []
for i, ID in enumerate(model_runs):

    df_chi = df_chi_all[i]
    quant = np.quantile(df_chi['drainage_area'], 0.2)
    df_chi1 = df_chi.loc[df_chi['drainage_area']>quant]

    m_sp = 0.5
    n_sp = 1.0
    # n_sp = df_params['n_sp'].loc[ID]
    U = df_params['U'].loc[ID]

    Ksp = U/np.mean(df_chi1['m_chi'])**n_sp
    # Qstar = df_results['Q/P'].iloc[i]
    Qstar = 1

    K_all.append(Ksp/Qstar)
    Ksp_all.append(Ksp)
K_all = np.array(K_all)
Ksp_all = np.array(Ksp_all)

# %% calculate with linear diffusion correction factor

Ksp_corr_all = []
K_corr_all = []
diff_all = []
Lh_all = []
for i, ID in enumerate(model_runs):
    df_chi = df_chi_all[i]
    quant = np.quantile(df_chi['drainage_area'], 0.2)
    df_chi1 = df_chi.loc[df_chi['drainage_area']>quant]

    m_sp = 0.5
    # n_sp = df_params['n_sp'].iloc[i]
    n_sp = 1
    U = df_params['U'].loc[ID]

    df_ht = df_ht_all[i]
    Lh = df_ht['Lh'].median()
    diff = 2*(Lh * U)/df_params['v0'].loc[ID]

    Ksp = (U+diff)/np.mean(df_chi1['m_chi'])**n_sp
    # Qstar = df_results['Q/P'].loc[ID]
    Qstar = 1

    K_corr_all.append(Ksp/Qstar)
    Ksp_corr_all.append(Ksp)
    diff_all.append(diff)
    Lh_all.append(Lh)

K_corr_all = np.array(K_corr_all)
Ksp_corr_all = np.array(Ksp_corr_all)
diff_all = np.array(diff_all)
Lh_all = np.array(Lh_all)
# %%

tsc = 3600 * 24 * 365
plt.figure()
plt.scatter(df_params['K'].loc[model_runs]*tsc, K_all*tsc)
plt.scatter(df_params['K'].loc[model_runs]*tsc, K_corr_all*tsc, c=Lh_all)
plt.axline([0,0],[1e-6,1e-6], linestyle='--', color='k')

# %%
