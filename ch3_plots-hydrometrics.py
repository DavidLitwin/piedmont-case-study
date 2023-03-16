#%%

import numpy as np
import pandas as pd
from datetime import timedelta
import statsmodels.api as sm

import matplotlib.pyplot as plt
from matplotlib import colors

import dataretrieval.nwis as nwis
from Hydrograph.hydrograph import sepBaseflow

# %% load precip

path_DR = "C:/Users/dgbli/Documents/Research/Soldiers Delight/data_processed/DruidRun_precip_15min_2022_4-2023_1.csv"
path_BR = "C:/Users/dgbli/Documents/Research/Oregon Ridge/data_processed/Baisman_precip_15min_2022_4-2023_1.csv"

dfp_DR = pd.read_csv(path_DR)
dfp_BR = pd.read_csv(path_BR)

dfp_BR['Date'] = pd.to_datetime(dfp_BR['Datetime'], utc=True)
dfp_BR.set_index('Date', inplace=True)
dfp_BR.drop(columns=['Datetime'], inplace=True)

dfp_DR['Date'] = pd.to_datetime(dfp_DR['Datetime'], utc=True)
dfp_DR.set_index('Date', inplace=True)
dfp_DR.drop(columns=['Datetime'], inplace=True)

# %% Baisman Run: Load Q

site_BR = '01583580'
site_PB = '01583570'

dfq = nwis.get_record(sites=site_BR, service='iv', start='2022-06-01', end='2023-01-26')
dfqug = nwis.get_record(sites=site_PB, service='iv', start='2022-06-01', end='2023-01-26')

# dfq.to_csv(path+'dfq.csv')
# dfqug.to_csv(path+'dfqug.csv')

#%% Baisman run: process Q

# area normalized discharge
area_BR = 381e4 #m2
dfq['Total runoff [m^3 s^-1]'] = dfq['00060']*0.3048**3 #m3/ft3 
dfq.drop(columns=['00060', 'site_no', '00065', '00065_cd'], inplace=True)

area_PB = 37e4 #m2
dfqug['Total runoff [m^3 s^-1]'] = dfqug['00060']*0.3048**3 #m3/ft3
dfqug.drop(columns=['00060', 'site_no', '00065', '00065_cd'], inplace=True)

# index from string to datetime
dfq['Date'] = pd.to_datetime(dfq.index, utc=True)
dfq.set_index('Date', inplace=True)

dfqug['Date'] = pd.to_datetime(dfqug.index, utc=True)
dfqug.set_index('Date', inplace=True)

#%% baseflow separation and events for Baisman

dfq_in = dfq.drop(columns='00060_cd')
dfq_in = dfq_in.resample('15min').mean()

dfq_BR = sepBaseflow(dfq_in, 15, area_BR*1e-6, k=0.000546, tp_min=4)

#%% merge precip

dfq_BR = dfq_BR.merge(dfp_BR, how='inner', on='Date')

# %% group data to events

# Peakflow volume [m^3]
dfq_BR['Qf'] = dfq_BR['Peakflow [m^3 s^-1]'] * 3600 * dfq_BR['dt [hour]'] *(1/area_BR) * 1000 # sec/hr * hr * 1/m2 * mm/m

dfe_BR = dfq_BR.groupby('Peak nr.').agg({'Peakflow starts': 'min', 
                                 'Peakflow ends': 'max',
                                 'Qf':'sum',
                                 'Max. flow [m^3 s^-1]':'max',
                                 'Baseflow [m^3 s^-1]':'min',
                                 })
dfe_BR['Precip starts'] = dfe_BR['Peakflow starts'] - timedelta(hours=6)
dfe_BR['Precip ends'] = dfe_BR['Peakflow ends'] - timedelta(hours=2)

# remove events where one ends before another begins
overlapped = []
for i in range(len(dfe_BR)-1):
    if dfe_BR['Precip starts'].iloc[i+1] < dfe_BR['Precip ends'].iloc[i]:
        overlapped.append(dfe_BR.index[i])

dfe_BR = dfe_BR.drop(overlapped)

# event precip
event_P = np.zeros(len(dfe_BR))
for i in range(len(dfe_BR)):
    event_P[i] = np.sum(dfq_BR['P (mm)'].loc[dfe_BR['Precip starts'].iloc[i]:dfe_BR['Precip ends'].iloc[i]])
dfe_BR['P'] = event_P

# remove where event precip is < 1 mm
dfe_BR = dfe_BR.drop(dfe_BR.index[dfe_BR['P'] < 1])

dfe_BR['Qb0'] = dfe_BR['Baseflow [m^3 s^-1]'] * 3600 * 24 * (1/area_BR) * 1000
#%% Event runoff ratio

fig, ax = plt.subplots()
sc = ax.scatter(dfe_BR['P'], dfe_BR['Qf'], c=dfe_BR['Qb0'], cmap='plasma', alpha=0.6)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel('Event Q (mm)')
ax.set_xlabel('Event P (mm)')
fig.colorbar(sc, label='Qb initial')
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/Event_RR_BR.png')
plt.show()

#%%
# plot baseflow and discharge with events

s = 8
P_plot = dfq_BR['P (mm)'] * 4 # mm/hr
Q_plot = dfq_BR['Total runoff [m^3 s^-1]'] * (1/area_BR) * 3600 * 1000 # 1/m2 sec/hr mm/m
Qb_plot = dfq_BR['Baseflow [m^3 s^-1]'] * (1/area_BR) * 3600 * 1000 # 1/m2 sec/hr mm/m


fig, ax = plt.subplots(figsize=(8,4))
ax.plot(Q_plot, 'k-')
ax.plot(Qb_plot, 'b-')
ax.scatter(dfe_BR['Peakflow starts'], 
           Qb_plot.loc[dfe_BR['Peakflow starts']],
           color='g', s=s, zorder=100)
ax.scatter(dfe_BR['Peakflow ends'], 
           Qb_plot.loc[dfe_BR['Peakflow ends']], 
           color='r', s=s, zorder=101)
ax.set_yscale('log')
ax.set_ylabel('Q (mm/hr)')

ax1 = ax.twinx()
ax1.plot(P_plot)
ax1.scatter(dfe_BR['Precip starts'], 
           P_plot.loc[dfe_BR['Precip starts']],
           color='g', s=s, zorder=102)
ax1.scatter(dfe_BR['Precip ends'], 
           P_plot.loc[dfe_BR['Precip ends']], 
           color='r', s=s, zorder=103)
ax1.set_ylim(2*P_plot.max(), 0)
ax1.set_ylabel('P (mm/hr)')
fig.autofmt_xdate()
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/BR_Q_P.png', transparent=True)
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/BR_Q_P.pdf', transparent=True)

# %% load Druids discharge

path = "C:/Users/dgbli/Documents/Research/Soldiers Delight/data_processed/DruidRun_discharge_15min_2022_3-2022_9.csv"
dfq_DR = pd.read_csv(path)

dfq_DR['Date'] = pd.to_datetime(dfq_DR['Datetime'], utc=True)
dfq_DR.set_index('Date', inplace=True)
dfq_DR.drop(columns=['Datetime'], inplace=True)

area_DR = 107e4 #m2
dfq_DR['Total runoff [m^3 s^-1]'] = dfq_DR['Q m3/s']
dfq_DR.drop(columns=['Q m3/s'], inplace=True)

# %%

dfq_DR = sepBaseflow(dfq_DR, 15, area_DR*1e-6, k=0.000546, tp_min=4)

#%% merge precip

dfq_DR = dfq_DR.merge(dfp_DR, how='inner', on='Date')

# %% group data to events

# Peakflow volume [m^3]
dfq_DR['Qf'] = dfq_DR['Peakflow [m^3 s^-1]'] * 3600 * dfq_DR['dt [hour]'] *(1/area_DR) * 1000 # sec/hr * hr * 1/m2 * mm/m

dfe_DR = dfq_DR.groupby('Peak nr.').agg({'Peakflow starts': 'min', 
                                 'Peakflow ends': 'max',
                                 'Qf':'sum',
                                 'Max. flow [m^3 s^-1]':'max',
                                 'Baseflow [m^3 s^-1]':'min',
                                 })
dfe_DR['Precip starts'] = dfe_DR['Peakflow starts'] - timedelta(hours=2)
dfe_DR['Precip ends'] = dfe_DR['Peakflow ends'] - timedelta(hours=1)

# remove events where one ends before another begins
overlapped = []
for i in range(len(dfe_DR)-1):
    if dfe_DR['Precip starts'].iloc[i+1] < dfe_DR['Precip ends'].iloc[i]:
        overlapped.append(dfe_DR.index[i])

dfe_DR = dfe_DR.drop(overlapped)

# event precip
event_P = np.zeros(len(dfe_DR))
for i in range(len(dfe_DR)):
    event_P[i] = np.sum(dfq_DR['P (mm)'].loc[dfe_DR['Precip starts'].iloc[i]:dfe_DR['Precip ends'].iloc[i]])
dfe_DR['P'] = event_P

# remove where event precip is < 1 mm
dfe_DR = dfe_DR.drop(dfe_DR.index[dfe_DR['P'] < 1])

dfe_DR['Qb0'] = dfe_DR['Baseflow [m^3 s^-1]'] * 3600 * 24 * (1/area_DR) * 1000

#%% plot baseflow and discharge with events

s = 8
P_plot = dfq_DR['P (mm)'] * 4 # mm/hr
Q_plot = dfq_DR['Total runoff [m^3 s^-1]'] * (1/area_DR) * 3600 * 1000 # 1/m2 sec/hr mm/m
Qb_plot = dfq_DR['Baseflow [m^3 s^-1]'] * (1/area_DR) * 3600 * 1000 # 1/m2 sec/hr mm/m


fig, ax = plt.subplots(figsize=(8,4))
ax.plot(Q_plot, 'k-')
ax.plot(Qb_plot, 'b-')
ax.scatter(dfe_DR['Peakflow starts'], 
           Qb_plot.loc[dfe_DR['Peakflow starts']],
           color='g', s=s, zorder=100)
ax.scatter(dfe_DR['Peakflow ends'], 
           Qb_plot.loc[dfe_DR['Peakflow ends']], 
           color='r', s=s, zorder=101)
ax.set_yscale('log')
ax.set_ylabel('Q (mm/hr)')

ax1 = ax.twinx()
ax1.plot(P_plot)
ax1.scatter(dfe_DR['Precip starts'], 
           P_plot.loc[dfe_DR['Precip starts']],
           color='g', s=s, zorder=102)
ax1.scatter(dfe_DR['Precip ends'], 
           P_plot.loc[dfe_DR['Precip ends']], 
           color='r', s=s, zorder=103)
ax1.set_ylim(2*P_plot.max(), 0)
ax1.set_ylabel('P (mm/hr)')
fig.autofmt_xdate()
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/DR_Q_P.png', transparent=True)
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/DR_Q_P.pdf', transparent=True)

#%% Event runoff ratio: Druids

# fig, axs = plt.subplots()
# sc = ax.scatter(dfe_DR['P'], dfe_DR['Qf'], c=dfe_DR['Qb0'], cmap='plasma', alpha=0.6)
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_ylabel('Event Q (mm)')
# ax.set_xlabel('Event P (mm)')
# fig.colorbar(sc, label='Qb initial')
# plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/Event_RR_DR.png')
# plt.show()

# %% All without regression

fig, axs = plt.subplots(ncols=2, figsize=(8,5))
axs[0].scatter(dfe_BR['P'], dfe_BR['Qf'], c=dfe_BR['Qb0'], cmap='plasma', alpha=0.6)
axs[0].axline([0,0], [1,1], color='k', linestyle='--')
axs[0].set_ylim((0.005,60))
axs[0].set_xlim((1,110))
axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].set_ylabel('Event Q (mm)')
axs[0].set_xlabel('Event P (mm)')
axs[0].set_title('Baisman Run')

axs[1].scatter(dfe_DR['P'], dfe_DR['Qf'], c=dfe_DR['Qb0'], cmap='plasma', alpha=0.6)
axs[1].axline([0,0], [1,1], color='k', linestyle='--')
axs[1].set_ylim((0.005,60))
axs[1].set_xlim((1,110))
axs[1].set_yscale('log')
axs[1].set_xscale('log')
axs[1].set_ylabel('Event Q (mm)')
axs[1].set_xlabel('Event P (mm)')
axs[1].set_title('Druids Run')
# plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/Event_RR.png')
plt.show()

# %% regression for event runoff

dfe_BR = dfe_BR.sort_values(by=['P'])
# regression
X = np.log(dfe_BR['P'].values)
X = sm.add_constant(X)
y = np.log(dfe_BR['Qf'].values)

model_BR = sm.OLS(y, X)
results_BR = model_BR.fit()
# print(results_BR.summary())

dfe_DR = dfe_DR.sort_values(by=['P'])
X = np.log(dfe_DR['P'].values)
X = sm.add_constant(X)
y = np.log(dfe_DR['Qf'].values)

model_DR = sm.OLS(y, X)
results_DR = model_DR.fit()
# print(results_DR.summary())

#%% runoff events with regression

fig, axs = plt.subplots(ncols=2, constrained_layout=True, figsize=(8,4))

pred_ols = results_BR.get_prediction()
iv_l = pred_ols.summary_frame()["obs_ci_lower"]
iv_u = pred_ols.summary_frame()["obs_ci_upper"]
fit = results_BR.fittedvalues

vmax = max([dfe_BR['Qb0'].max(), dfe_DR['Qb0'].max()])
vmin = max([dfe_BR['Qb0'].min(), dfe_DR['Qb0'].min()])

axs[1].scatter(dfe_BR['P'], dfe_BR['Qf'], c=dfe_BR['Qb0'], cmap='plasma', vmin=vmin, vmax=vmax, alpha=0.6)
axs[1].plot(dfe_BR['P'], np.exp(fit), "b-", alpha=0.5)
axs[1].fill_between(dfe_BR['P'], np.exp(iv_l) , np.exp(iv_u), alpha=0.15, color='gray')
axs[1].axline([0,0], [1,1], color='k', linestyle='--')
axs[1].text(0.1, 
            0.9, 
            # r'$r^2 = %.2f$'%results_BR.rsquared,
            r'$a = %.3f \pm %.3f$'%(results_BR.params[1], results_BR.bse[1]), 
            transform=axs[1].transAxes
            )
axs[1].set_ylim((0.005,60))
axs[1].set_xlim((2,110))
axs[1].set_yscale('log')
axs[1].set_xscale('log')
axs[1].set_ylabel('Event Q (mm)')
axs[1].set_xlabel('Event P (mm)')
axs[1].set_title('Baisman Run')
# axs[1].set_aspect(0.6)

pred_ols = results_DR.get_prediction()
iv_l = pred_ols.summary_frame()["obs_ci_lower"]
iv_u = pred_ols.summary_frame()["obs_ci_upper"]
fit = results_DR.fittedvalues

sc = axs[0].scatter(dfe_DR['P'], dfe_DR['Qf'], c=dfe_DR['Qb0'], cmap='plasma', vmin=vmin, vmax=vmax, alpha=0.6)
axs[0].plot(dfe_DR['P'], np.exp(fit), "b-", alpha=0.5)
axs[0].fill_between(dfe_DR['P'], np.exp(iv_l) , np.exp(iv_u), alpha=0.15, color='gray')
axs[0].axline([0,0], [1,1], color='k', linestyle='--')
axs[0].text(0.1, 
            0.9, 
            # r'$r^2 = %.2f$'%results_DR.rsquared,
            r'$a = %.3f \pm %.3f$'%(results_DR.params[1], results_DR.bse[1]), 
            transform=axs[0].transAxes
            )
axs[0].set_ylim((0.005,60))
axs[0].set_xlim((2,110))
axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].set_ylabel('Event Q (mm)')
axs[0].set_xlabel('Event P (mm)')
axs[0].set_title('Druids Run')
# axs[0].set_aspect(0.6)
# cax = axs[1].inset_axes([1.04, 0.1, 0.05, 0.8])
fig.colorbar(sc, ax=axs, label='Initial Baseflow (mm/d)')

plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/Event_RR.png', transparent=True)
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/Event_RR.pdf', transparent=True)

df_rr_reg = pd.DataFrame(data=[[results_DR.params[1], results_DR.bse[1], results_DR.params[0], results_DR.bse[0], results_DR.rsquared],
                               [results_BR.params[1], results_BR.bse[1], results_BR.params[0], results_BR.bse[0], results_BR.rsquared]],
                               columns=['exp', 'exp_err', 'coeff', 'coeff_err', 'rsquared'], index=['DR','BR']
                               )
df_rr_reg.to_csv('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/df_runoff_ratio_stats.csv', float_format="%.3f")


# %% O'loughlin analysis


fig, axs = plt.subplots(ncols=2, constrained_layout=True, figsize=(6,3))
axs[0].scatter(1/dfe_DR['Qb0'], dfe_DR['Qf']/dfe_DR['P'], c=dfe_DR['P'], norm=colors.LogNorm(vmin=4, vmax=100))
axs[0].set_xscale('log')
axs[0].set_yscale('log')
# axs[0].set_ylim((-0.01,0.5))
# axs[0].set_xlim((0.5,2.5))
axs[0].set_ylabel('Event Runoff Ratio (-)')
axs[0].set_xlabel(r'$1/Q_0$ $(mm/d)^{-1}$')
axs[0].set_title('Druids Run')
# axs[0].set_aspect(1)

sc = axs[1].scatter(1/dfe_BR['Qb0'], dfe_BR['Qf']/dfe_BR['P'], c=dfe_BR['P'], norm=colors.LogNorm(vmin=4, vmax=100))
axs[1].set_xscale('log')
axs[1].set_yscale('log')
# axs[1].set_ylim((-0.01,0.5))
# axs[1].set_xlim((0.5,2.5))
axs[1].set_ylabel('Event Runoff Ratio (-)')
axs[1].set_xlabel(r'$1/Q_0$ $(mm/d)^{-1}$')
axs[1].set_title('Baisman Run')

# axs[1].set_aspect(1)

fig.colorbar(sc, label='Event Precipitation (mm)')
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/Event_RR_Q0.png', transparent=True)
plt.savefig('C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/Event_RR_Q0.pdf', transparent=True)
# %%
