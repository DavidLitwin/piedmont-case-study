# -*- coding: utf-8 -*-
"""
Analyze dilution gaging data from Druid Run and Druid Run Upper Gage.


Created on Fri Mar 25 11:32:02 2022

@author: dgbli
"""
#%%
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from dilution_gaging import plot_timeseries, calc_discharge

path = '/Users/dlitwin/Documents/Research/Soldiers Delight/data/pressure_transducer_gaging'
path1 = '/Users/dlitwin/Documents/Research/Soldiers Delight/data_processed'
figpath = '/Users/dlitwin/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/'

discharge_DR = {}
stage_DR = {}
discharge_UG = {}

#%% March 3, 2022 Druids Run

# file = os.path.join(path,'DR_20220304.csv') 
# file_mass = os.path.join(path,'DR_20220304_mass.csv') 

# plot_timeseries(file)

# start_time = pd.Timestamp('2022-03-04 13:30:00')
# end_time = pd.Timestamp('2022-03-04 15:30:00')

# start_time_inj = pd.Timestamp('2022-03-04 13:40:00')
# end_time_inj = pd.Timestamp('2022-03-04 14:50:00')

# Q = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj, mass_index='all')
# print('Discharge: %.3f L/s'%Q)
# discharge_DR[start_time_inj] = Q
# stage_DR[start_time_inj] = np.nan

#%% March 11, 2022 Druids Run

file = os.path.join(path,'DR_20220311.csv') 
file_mass = os.path.join(path,'DR_20220311_mass.csv') 

# plot_timeseries(file)

start_time = pd.Timestamp('2022-03-11 10:52:00')
end_time = pd.Timestamp('2022-03-11 12:24:00')

start_time_inj = pd.Timestamp('2022-03-11 10:57:00')
end_time_inj = pd.Timestamp('2022-03-11 12:00:00')

Q = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj)
print('Discharge: %.3f L/s'%Q)
discharge_DR[start_time_inj] = Q
stage_DR[start_time_inj] = -25.4 # cm from top of PVC

#%% March 24, 2022 Druids Run

file = os.path.join(path,'DR_20220324.csv') 
file_mass = os.path.join(path,'DR_20220324_mass.csv') 

# plot_timeseries(file)

start_time = pd.Timestamp('2022-03-24 10:18:00')
end_time = pd.Timestamp('2022-03-24 11:28:00')

start_time_inj = pd.Timestamp('2022-03-24 10:22:00')
end_time_inj = pd.Timestamp('2022-03-24 11:24:00')

Q = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj)
print('Discharge: %.3f L/s'%Q)
discharge_DR[start_time_inj] = Q
stage_DR[start_time_inj] = -23.8 # cm from top of PVC

#%% March 11, 2022 Upper WS

file = os.path.join(path,'DRupperWS_20220311.csv') 
file_mass = os.path.join(path,'DRupperWS_20220311_mass.csv') 

# plot_timeseries(file)

start_time = pd.Timestamp('2022-03-11 08:30:00')
end_time = pd.Timestamp('2022-03-11 09:53:00')

start_time_inj = pd.Timestamp('2022-03-11 08:35:00')
end_time_inj = pd.Timestamp('2022-03-11 09:50:00')

Q = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj, mass_index=0)
print('Discharge: %.3f L/s'%Q)
discharge_UG[start_time_inj] = Q

#%% March 24, 2022 Upper WS

file = os.path.join(path,'DRupperWS_20220324.csv') 
file_mass = os.path.join(path,'DRupperWS_20220324_mass.csv') 

plot_timeseries(file)

start_time = pd.Timestamp('2022-03-24 11:49:00')
end_time = pd.Timestamp('2022-03-24 12:55:00')

start_time_inj = pd.Timestamp('2022-03-24 11:55:00')
end_time_inj = pd.Timestamp('2022-03-24 12:54:00')

Q = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj, mass_index=0)
print('Discharge: %.3f L/s'%Q)
discharge_UG[start_time_inj] = Q

#%% April 19, 2022 Druids Run

file = os.path.join(path,'DR_20220419.csv') 
file_mass = os.path.join(path,'DR_20220419_mass.csv') 

plot_timeseries(file)

start_time = pd.Timestamp('2022-04-19 08:12:00')
end_time = pd.Timestamp('2022-04-19 09:05:00')

start_time_inj = pd.Timestamp('2022-04-19 08:15:00')
end_time_inj = pd.Timestamp('2022-04-19 08:50:00')

Q = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj)
print('Discharge: %.3f L/s'%Q)
discharge_DR[start_time_inj] = Q
stage_DR[start_time_inj] = -14.2 # cm from top of PVC

#%% April 19, 2022 Upper WS

file = os.path.join(path,'DRupperWS_20220419.csv') 
file_mass = os.path.join(path,'DRupperWS_20220419_mass.csv') 

plot_timeseries(file)

start_time = pd.Timestamp('2022-04-19 09:28:00')
end_time = pd.Timestamp('2022-04-19 11:04:00')

start_time_inj = pd.Timestamp('2022-04-19 09:31:00')
end_time_inj = pd.Timestamp('2022-04-19 09:50:00')

Q = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj, mass_index=0)
print('Discharge: %.3f L/s'%Q)
discharge_UG[start_time_inj] = Q

#%% April 27, 2022 Upper WS

file = os.path.join(path,'DRupperWS_20220427.csv') 
file_mass = os.path.join(path,'DRupperWS_20220427_mass.csv') 

plot_timeseries(file)

start_time = pd.Timestamp('2022-04-27 10:28:00')
end_time = pd.Timestamp('2022-04-27 11:50:00')

start_time_inj = pd.Timestamp('2022-04-27 10:30:00')
end_time_inj = pd.Timestamp('2022-04-27 11:20:00')

Q = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj, mass_index=0)
print('Discharge: %.3f L/s'%Q)
discharge_UG[start_time_inj] = Q

#%% May 7, 2022 Druids Run

start_time = pd.Timestamp('2022-05-07 10:54:00')
discharge_DR[start_time] = 1.720e3 #L/s from EM sensor
stage_DR[start_time] = 22.5 # cm from top of PVC


#%% June 28, 2022 Druids Run

# missing??

#%% August 16, 2022 Druids Run

file = os.path.join(path,'DR_20220816.csv') 
file_mass = os.path.join(path,'DR_20220816_mass.csv') 

plot_timeseries(file)

start_time = pd.Timestamp('2022-08-16 09:19:00')
end_time = pd.Timestamp('2022-08-16 09:55:00')

start_time_inj = pd.Timestamp('2022-08-16 09:21:00')
end_time_inj = pd.Timestamp('2022-08-16 09:45:00')

Q = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj)
print('Discharge: %.3f L/s'%Q)
discharge_DR[start_time_inj] = Q
stage_DR[start_time_inj] = -28.7 # cm from top of PVC


#%% September 6, 2022 Druids Run

start_time = pd.Timestamp('2022-09-06 09:30:00') # average of 9:19 start, 9:41 finish
discharge_DR[start_time] = 0.199e3 #L/s from EM sensor (20220906.tsv)
stage_DR[start_time] = -2.0 # cm from top of PVC (average of 0 cm at start, -4 cm below at finish)


start_time = pd.Timestamp('2022-09-06 10:25:00') # average of 10:14 start, 10:36 finish
discharge_DR[start_time] = 0.129e3 #L/s from EM sensor (2022096B_recalc.tsv)
stage_DR[start_time] = -8.0 # cm from top of PVC (average of -7 cm at start, -9 cm below at finish)


start_time = pd.Timestamp('2022-09-06 10:52:00') # average of 10:41 start, 11:03 finish
discharge_DR[start_time] = 0.103e3 #L/s from EM sensor (2022096C.tsv)
stage_DR[start_time] = -10.8 # cm from top of PVC (average of -10.5 cm at start, -11.1 cm below at finish)

#%% December 8, 2022 Druids Run

file = os.path.join(path,'DR_20221208.csv') 
file_mass = os.path.join(path,'DR_20221208_mass.csv') 

plot_timeseries(file)

# first injection
start_time = pd.Timestamp('2022-12-08 09:35:00')
end_time = pd.Timestamp('2022-12-08 10:11:00')

start_time_inj = pd.Timestamp('2022-12-08 09:40:00')
end_time_inj = pd.Timestamp('2022-12-08 10:06:00')

Q1 = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj, mass_index=0)
print('Discharge: %.3f L/s'%Q1)

# second injection
start_time = pd.Timestamp('2022-12-08 10:09:00')
end_time = pd.Timestamp('2022-12-08 10:50:00')

start_time_inj = pd.Timestamp('2022-12-08 10:12:00')
end_time_inj = pd.Timestamp('2022-12-08 10:45:00')

Q2 = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj, mass_index=1)
print('Discharge: %.3f L/s'%Q2)

discharge_DR[start_time_inj] = (Q1 + Q2)/2
stage_DR[start_time_inj] = -24.2 # cm from top of PVC

#%% December 8, 2022 upstream tributaries

file = os.path.join(path,'DR_20221208.csv') 
file_mass = os.path.join(path,'Tribs_20221208_mass.csv') 

plot_timeseries(file)

# Tributary 1 (Southern)
start_time = pd.Timestamp('2022-12-08 11:10:00')
end_time = pd.Timestamp('2022-12-08 11:45:00')

start_time_inj = pd.Timestamp('2022-12-08 11:14:00')
end_time_inj = pd.Timestamp('2022-12-08 11:40:00')

Q = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj, mass_index=0)
print('Southern Trib Discharge: %.3f L/s'%Q)

# Tributary 2 (Northeastern)
start_time = pd.Timestamp('2022-12-08 12:48:00')
end_time = pd.Timestamp('2022-12-08 13:15:00')

start_time_inj = pd.Timestamp('2022-12-08 12:50:00')
end_time_inj = pd.Timestamp('2022-12-08 13:10:00')

Q = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj, mass_index=0)
print('Northeastern Trib Discharge: %.3f L/s'%Q)


#%% January 26, 2023 Druids Run

file = os.path.join(path,'DR_20230126.csv') 
file_mass = os.path.join(path,'DR_20230126_mass.csv') 

plot_timeseries(file)

start_time = pd.Timestamp('2023-01-26 09:16:00')
end_time = pd.Timestamp('2023-01-26 09:42:00')

start_time_inj = pd.Timestamp('2023-01-26 09:19:00')
end_time_inj = pd.Timestamp('2023-01-26 09:40:00')

Q1 = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj, mass_index=0)
print('Discharge: %.3f L/s'%Q1)


start_time = pd.Timestamp('2023-01-26 09:40:00')
end_time = pd.Timestamp('2023-01-26 10:20:00')

start_time_inj = pd.Timestamp('2023-01-26 09:43:00')
end_time_inj = pd.Timestamp('2023-01-26 10:10:00')

Q2 = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj, mass_index=1)
print('Discharge: %.3f L/s'%Q2)


discharge_DR[start_time_inj] = (Q1+Q2)/2
stage_DR[start_time_inj] = -14.0 # cm from top of PVC

#%% January 26 Upper WS

file = os.path.join(path,'DRupperWS_20230126.csv') 
file_mass = os.path.join(path,'DRupperWS_20230126_mass.csv') 

plot_timeseries(file)

start_time = pd.Timestamp('2023-01-26 10:54:00')
end_time = pd.Timestamp('2023-01-26 11:50:00')

start_time_inj = pd.Timestamp('2023-01-26 10:57:00')
end_time_inj = pd.Timestamp('2023-01-26 11:30:00')

Q = calc_discharge(file, file_mass, start_time, end_time, start_time_inj, end_time_inj)
print('Discharge: %.3f L/s'%Q)

discharge_UG[start_time_inj] = Q



#%% plot rating curve
def power_law(x, a, b, c):
    return a*np.power(x, b) + c

df1 = pd.DataFrame.from_dict(stage_DR, orient='index',columns=['Stage'])
df2 = pd.DataFrame.from_dict(discharge_DR, orient='index',columns=['Q'])
df = df1.join(df2)
df['Stage offset'] = (df['Stage'] + 100)/100


plt.figure()
plt.scatter(df['Stage offset'], df['Q'])


pars, cov = curve_fit(f=power_law, xdata=df['Stage offset'], ydata=df['Q'], p0=[0.1, 2, 1], bounds=(-1000, 1000))
stdevs = np.sqrt(np.diag(cov))

#%%
  
xp = np.linspace(0,3, 100)
plt.figure(figsize=(5,4))
plt.scatter(df['Stage offset'], df['Q'], label='measured pts')
plt.plot(xp, power_law(xp, pars[0], pars[1], pars[2]), 'k--', label=r'fit $%.3f x^{%.3f} + %.3f$'%(pars[0], pars[1], pars[2]))
plt.xlim((0.4,1.6))
plt.ylim((3,3000))
plt.xscale('log')
plt.yscale('log')
plt.legend(frameon=False)
plt.xlabel('Stage (m)')
plt.ylabel('discharge (L/s)')
plt.tight_layout()
plt.savefig(figpath+'rating_curve_DR.pdf', transparent=True)
plt.savefig(figpath+'rating_curve_DR.png', transparent=True, dpi=300)

#%% export data

pickle.dump(df,open(os.path.join(path1, 'rating_pts.p'), 'wb'))
pickle.dump(pars,open(os.path.join(path1, 'rating_exp_coeffs.p'), 'wb'))

pickle.dump(discharge_DR,open(os.path.join(path1,'discharge_DR.p'), 'wb'))
pickle.dump(discharge_UG,open(os.path.join(path1,'discharge_UG.p'), 'wb'))
# %%
