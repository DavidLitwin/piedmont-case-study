#%%

import numpy as np
import pandas as pd
import pickle
import glob
import rasterio as rd
import statsmodels.api as sm

import matplotlib.pyplot as plt
from matplotlib import colors

#%%

path_DR = "C:/Users/dgbli/Documents/Research/Soldiers Delight/data_processed/Gianni_event_DR/"
path_BR = "C:/Users/dgbli/Documents/Research/Oregon Ridge/data_processed/Gianni_event_BAIS/"

file_DR = 'Druids Run.csv'
file_BR = 'Baisman Run.csv'
df_DR = pd.read_csv(path_DR+file_DR)
df_BR = pd.read_csv(path_BR+file_BR)

df_DR['date'] = pd.to_datetime(df_DR[['year', 'month', 'day', 'hour', 'minute', 'second']])
df_BR['date'] = pd.to_datetime(df_BR[['year', 'month', 'day', 'hour', 'minute', 'second']])
df_DR.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)
df_BR.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)


# event_names = ['Q_start', 'Q_end', 'P_start', 'P_end']
file = 'StartFinishFlowStartFinishRain.csv'
df_DR_event = pd.read_csv(path_DR+file) #, names=event_names
df_BR_event = pd.read_csv(path_BR+file)

# %%

fig, ax = plt.subplots()
ax.plot(df_DR.date, df_DR['flow (mm/day)'], 'k-')
ax.plot(df_DR.date, df_DR['baseflow (mm/day)'], 'b-')
ax.scatter(df_DR['date'].iloc[df_DR_event['Q_start']], df_DR['flow (mm/day)'].iloc[df_DR_event['Q_start'].values], c='g')
ax.scatter(df_DR['date'].iloc[df_DR_event['Q_end']], df_DR['flow (mm/day)'].iloc[df_DR_event['Q_end'].values], c='r')
ax.set_yscale('log')
ax.set_ylabel('Q (mm/day)')
ax1 = ax.twinx()
ax1.plot(df_DR.date, df_DR['rain (mm/day)'])
ax1.set_ylim(2*np.max(df_DR['rain (mm/day)']), 0)
ax1.set_ylabel('P (mm/day)')
fig.autofmt_xdate()
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/DR_Q_P.png')
plt.show()

#%%
fig, ax = plt.subplots()
ax.plot(df_BR.date, df_BR['flow (mm/day)'], 'k-')
ax.plot(df_BR.date, df_BR['baseflow (mm/day)'], 'b-')
ax.scatter(df_BR['date'].iloc[df_BR_event['Q_start']], df_BR['flow (mm/day)'].iloc[df_BR_event['Q_start'].values], c='g')
ax.scatter(df_BR['date'].iloc[df_BR_event['Q_end']], df_BR['flow (mm/day)'].iloc[df_BR_event['Q_end'].values], c='r')
ax.set_yscale('log')
ax.set_ylabel('Q (mm/day)')
ax1 = ax.twinx()
ax1.plot(df_BR.date, df_BR['rain (mm/day)'])
ax1.set_ylim(2*np.max(df_BR['rain (mm/day)']), 0)
ax1.set_ylabel('P (mm/day)')
fig.autofmt_xdate()
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/BR_Q_P.png')
plt.show()


# %%

df_BR['quickflow (mm)'] = (df_BR['flow (mm/day)'] - df_BR['baseflow (mm/day)'])/(24*4) # convert from mm/day to mm per time interval (15 min)
df_DR['quickflow (mm)'] = (df_DR['flow (mm/day)'] - df_DR['baseflow (mm/day)'])/(24*4)
df_BR['precip (mm)'] = df_BR['rain (mm/day)']/(24*4)
df_DR['precip (mm)'] = df_DR['rain (mm/day)']/(24*4)

df_BR_event['P'] = np.nan
df_BR_event['Qf'] = np.nan
df_BR_event['start_date'] = np.nan
for i in df_BR_event.index:
    df_BR_event['P'].iloc[i] = df_BR['precip (mm)'].iloc[df_BR_event['P_start'][i]:df_BR_event['P_end'][i]].sum()
    df_BR_event['Qf'].iloc[i] = df_BR['quickflow (mm)'].iloc[df_BR_event['Q_start'][i]:df_BR_event['Q_end'][i]].sum()
    df_BR_event['start_date'].iloc[i] = df_BR['date'].iloc[df_BR_event['P_start']]


df_DR_event['P'] = np.nan
df_DR_event['Qf'] = np.nan
df_DR_event['start_date'] = np.nan
for i in df_DR_event.index:
    df_DR_event['P'].iloc[i] = df_DR['precip (mm)'].iloc[df_DR_event['P_start'][i]:df_DR_event['P_end'][i]].sum()
    df_DR_event['Qf'].iloc[i] = df_DR['quickflow (mm)'].iloc[df_DR_event['Q_start'][i]:df_DR_event['Q_end'][i]].sum()
    df_DR_event['start_date'].iloc[i] = df_DR['date'].iloc[df_DR_event['P_start']]


#%%

# all
fig, axs = plt.subplots(ncols=2, figsize=(8,5))
axs[0].scatter(df_BR_event['P'], df_BR_event['Qf'], alpha=0.5, c='b')
axs[0].axline([0,0], [1,1], color='k', linestyle='--')
axs[0].set_ylim((-1,50))
axs[0].set_xlim((-3,110))
axs[0].set_ylabel('Event Q (mm)')
axs[0].set_xlabel('Event P (mm)')
axs[0].set_title('Baisman Run')

axs[1].scatter(df_DR_event['P'], df_DR_event['Qf'], alpha=0.5, c='r')
axs[1].axline([0,0], [1,1], color='k', linestyle='--')
axs[1].set_ylim((-1,50))
axs[1].set_xlim((-3,110))
axs[1].set_ylabel('Event Q (mm)')
axs[1].set_xlabel('Event P (mm)')
axs[1].set_title('Druids Run')
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/Event_RR.png')
plt.show()

# %%
