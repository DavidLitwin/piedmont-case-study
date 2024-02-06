# -*- coding: utf-8 -*-
"""
Clean stage timeseries and apply the rating curve to make discharge timeseries

Created on Fri Aug 26 10:00:57 2022

@author: dgbli
"""
#%%
import os
from copy import copy
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime

path = '/Users/dlitwin/Documents/Research/Soldiers Delight/data/pressure_transducer_gaging'
path1 = '/Users/dlitwin/Documents/Research/Soldiers Delight/data_processed'

#%% load data

# load rating from analyze_dilution_gaging.py
dfr = pickle.load(open(os.path.join(path1,'rating_pts.p'), 'rb'))
pars = pickle.load(open(os.path.join(path1,'rating_exp_coeffs.p'), 'rb'))


# load compensated pressure transducer data
file = 'DruidRunCulvert_20220408_compensated.csv'
dfA = pd.read_csv(os.path.join(path,file), header=11, parse_dates=[[0,1]])
dfA.set_index('Date_Time', inplace=True)

file = 'DruidRunCulvert_20220816_compensated.csv'
dfB = pd.read_csv(os.path.join(path,file), header=11, parse_dates=[[0,1]])
dfB.set_index('Date_Time', inplace=True)


file = 'DruidRunCulvert_20221022_compensated.csv'
dfC = pd.read_csv(os.path.join(path,file), header=11, parse_dates=[[0,1]])

# corrections
dfC['LEVEL'].iloc[4544:4565] -= 0.121
dfC['LEVEL'].iloc[6848:6890] = np.nan
dfC['LEVEL'].iloc[7082:7189] = np.nan
dfC['LEVEL'].iloc[8234:8342] = np.nan

dfC.set_index('Date_Time', inplace=True)
dfC['LEVEL'] = dfC['LEVEL'].interpolate(method='linear')

#%% clean and combine dataframes

plt.figure()
plt.plot(dfA.index, dfA['LEVEL'], 'k-')
plt.plot(dfB.index, dfB['LEVEL'], 'b-')
plt.plot(dfC.index, dfC['LEVEL'], 'r-')

# reindex dfA to even 5 minute intervals
desired_index = pd.date_range(start=datetime(2022,2,28,13,5), end=datetime(2022,4,8,13,15), freq="5min") 

# ht https://stackoverflow.com/a/52701851/11627361
dfA = (dfA
 .reindex(dfA.index.union(desired_index))
 .interpolate(method='time')
 .reindex(desired_index)
)

# indices to drop based on notes of when logger was out of water
NODATA = np.nan

dfA = dfA[dfA.index>=datetime(2022,2,28,13)]
dfA.loc[np.logical_and(dfA.index>=datetime(2022,4,1,9,50), dfA.index<datetime(2022,4,1,10,20)), 'LEVEL'] = NODATA
dfA = dfA[dfA.index<=datetime(2022,4,8,13, 15)]
dfA['LEVEL'] = dfA['LEVEL'].interpolate(method='linear')

dfB.loc[np.logical_and(dfB.index>=datetime(2022,5,10,14,30), dfB.index<datetime(2022,5,10,14,55)), 'LEVEL'] = NODATA
dfB.loc[np.logical_and(dfB.index>=datetime(2022,6,28,10,10), dfB.index<datetime(2022,6,28,10,45)), 'LEVEL'] = NODATA
dfB = dfB[dfB.index<=datetime(2022,8,16,9,15)]

dfC = dfC[dfC.index<=datetime(2022,9,14,20,0)]

# plt.figure()
# plt.plot(dfA.index, dfA['LEVEL'], 'k-')
# plt.plot(dfB.index, dfB['LEVEL'], 'b-')

# merge dataframes 
df = pd.concat([dfA, dfB, dfC])

#%% convert stage to adjusted stage rating curve uses

indices = [df.index.get_loc(ind, method='nearest') for ind in dfr.index]
wl_gage = df['LEVEL'].iloc[indices]

# confirm decent fit with slope = 1

plt.figure()
plt.scatter(dfr['Stage offset'], wl_gage)
plt.axline([min(dfr['Stage offset']), min(wl_gage)], slope=1, color='r')
plt.xlabel('Offset Measured Stage (m)')
plt.ylabel('Transducer Stage (m)')

# offset 
b = min(dfr['Stage offset']) - min(wl_gage)
df['level offset'] = df['LEVEL'] + b

# calculate discharge
def power_law(x, a, b, c):
    return a*np.power(x, b) + c

df['Q'] = power_law(df['level offset'].values, pars[0], pars[1], pars[2])

plt.figure()
plt.plot(df.index, df['Q'], 'k-', zorder=1)
plt.scatter(dfr.index, dfr['Q'], zorder=2)
plt.yscale('log')
# scatter the rating curve data
# color where stage is < 1 cm


cmap1 = copy(cm.plasma)
cmap1.set_bad('k')

# ht https://stackoverflow.com/a/36521456
def plot_colourline(x,y,c, cmap):
    c = cmap((c-np.min(c))/(np.max(c)-np.min(c)))
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=c[i])
    return

cl = df['LEVEL']
cl[cl>0.01] = np.nan
plt.figure(figsize=(10,6))
plot_colourline(df.index, df['Q'], cl, cmap1)
plt.scatter(dfr.index, dfr['Q'], zorder=2)
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Discharge (L/s)')
plt.savefig('DR_discharge.png', dpi=300)


#%% 

df['Q m3/s'] = df['Q']/1000

df['Q m3/s'].to_csv('DruidRun_discharge_2022_3-2022_9.csv', index_label='Datetime', float_format='%.4e')

df2 = df.resample('15min').mean()
df2['Q m3/s'].to_csv(os.path.join(path1,'DruidRun_discharge_15min_2022_3-2022_9.csv'), index_label='Datetime', float_format='%.4e')

plt.figure()
plt.plot(df2['Q m3/s'])
plt.yscale('log')