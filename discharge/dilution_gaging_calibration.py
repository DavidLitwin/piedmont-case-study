# -*- coding: utf-8 -*-
"""
Calibration of relationship between concentration and conductivity.

Created on Fri Mar 25 11:19:17 2022

@author: dgbli
"""
#%%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as spi
from sklearn.linear_model import LinearRegression

#%%
path = '/Users/dlitwin/Documents/Research/Soldiers Delight/data/dilution_gaging'
file_cal = 'WCB_20210616_Downstream_labcalibration.csv'
df_cal = pd.read_csv(os.path.join(path,file_cal), header=1, parse_dates=[1], index_col=0, usecols=range(5), infer_datetime_format=True)
df_cal.columns = ['time', 'cond low', 'cond full', 'temp']

# calc specific conductance
cond_to_spec_cond = lambda c, t: c/(1+0.02*(t-25)) # from Paige
df_cal['spec_cond'] = cond_to_spec_cond(df_cal['cond low'], df_cal['temp'])

# plot and have a look
plt.figure()
df_cal['spec_cond'].plot()

# calibtration procedure info
sample_vol = 500 #ml
vol_added = np.ones(9); vol_added[0] = 0 #ml
mass_added = 20*np.ones(9); mass_added[0] = 0 #mg/ml
total_vol = sample_vol + np.cumsum(vol_added)
total_mass = np.cumsum(mass_added)
concentration = total_mass/total_vol
starts = [50, 176, 275, 372, 456, 555, 655, 812, 1010]
ends = [131, 253, 345, 436, 536, 633, 783, 971, 1330]

# calculate averages for each concentration period
df_cal_out = pd.DataFrame({'C (g/L)': concentration})
spec_cond = []
for start, end in zip(starts, ends):
    spec_cond.append(df_cal['spec_cond'][start:end].mean())
df_cal_out['spec_cond'] = spec_cond

# linear regression on averaged periods
X = df_cal_out['spec_cond'].values.reshape((len(starts),1))
Y = df_cal_out['C (g/L)'].values.reshape((len(starts),1))
lr = LinearRegression().fit(X,Y)
fit_y = lr.predict(X)
coef, inter = lr.coef_[0][0], lr.intercept_[0]

# plot
plt.figure()
df_cal_out.plot.scatter('spec_cond', 'C (g/L)', label='Experimental')
plt.plot(X,fit_y, 'b--', label='Fit')
plt.legend()