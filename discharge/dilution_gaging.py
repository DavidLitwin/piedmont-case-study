# -*- coding: utf-8 -*-
"""
Functions for analysis of dilution gaging data

Created on Fri Mar 25 11:03:26 2022

@author: dgbli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as spi

#%%
def plot_timeseries(cond_file):
    """
    Plot conductivity and temperature from the csv file created by export
    from HOBOWare software using Pandas.

    Parameters
    ----------
    cond_file : str
        full path to timeseries csv file.

    Returns
    -------
    None.

    """
    
    # timeseries
    df = pd.read_csv(cond_file, header=1, parse_dates=[1], index_col=0, usecols=range(5), infer_datetime_format=True)
    df.columns = ['time', 'cond low', 'cond full', 'temp']
    df.set_index('time', inplace=True)
    df.plot(title='Cond/Temp Timeseries')


def calc_discharge(cond_file, mass_file, clip_start, clip_end, inj_start, inj_end, mass_index='all'):
    """
    Calculate discharge from conductivity file, mass file, and information
    on times for starting and stopping integration and interpolation.

    Parameters
    ----------
    cond_file : str
        Full path to timeseries csv file. 
    mass_file : str
        Full path to mass csv file, with header Time, m1, m2, where m1+m2 is 
        the total amount of salt injected in one injection. There are usually
        two values m1 and m2 because two bags of salt were added. If only one 
        was used make m2 0.0.
    clip_start : Pandas Timestamp
        Time that integration will start. Usually start of time that sensor
        is in water or after influence of previously injected tracer has left.
    clip_end : Pandas Timestamp
        Time integration will end. Usually right before sensor is pulled from
        water or before another injection occurs that is being excluded from
        integration.
    inj_start : Pandas Timestamp
        Time right before signal of tracer appears. Conductivity before this time
        will be considered background.
    inj_end : Pandas Timestamp
        Time after influence of tracer has disappeared. Conductivity after this
        time will be considered background
    mass_index : str or int, optional
        Injection to consider for discharge calcultions. The value should either
        be the string 'all', in which case all masses are added together,
        or it should be an integer equal to the index of tracer mass corresponding
        to the breakthrough curve isolated with clip_start and clip_end.
        The default is 'all'.

    Returns
    -------
    Q : float
        Discharge estimated by trapezoidal integration.

    """
    
    
    # timeseries
    df = pd.read_csv(cond_file, header=1, parse_dates=[1], index_col=0, usecols=range(5), infer_datetime_format=True)
    df.columns = ['time', 'cond low', 'cond full', 'temp']
    df.set_index('time', inplace=True)
    
    # masses 
    df_mass = pd.read_csv(mass_file, parse_dates=[0], infer_datetime_format=True)
    df_mass['m'] = df_mass.m1+df_mass.m2

    # mask to useable data
    mask = (df.index >= clip_start) & (df.index <= clip_end)
    df = df.loc[mask]
    
    # specific conductance
    cond_to_spec_cond = lambda c, t: c/(1+0.02*(t-25)) # from Paige
    df['spec_cond'] = cond_to_spec_cond(df['cond low'], df['temp'])
    
    # concentration conversion factor (from 2021 calibration)
    CF = 0.48686
    
    # create background timeseries from original timeseries by removing data when tracer is apparent and linearly interpolating in between these two points.
    df['background'] = df['spec_cond']
    mask = (df.index >= inj_start) & (df.index <= inj_end)
    df['background'].loc[mask] = np.nan
    df['background'].interpolate(method='linear', inplace=True)
    df['background'].loc[df['background']>df['spec_cond']] = df['spec_cond'].loc[df['background']>df['spec_cond']]
    
    # calculate as the conductance above background times the correction factor
    df['concentration'] = (df['spec_cond'] - df['background'])*CF
    
    if isinstance(mass_index, int):
        mass = df_mass.m.iloc[mass_index]
    elif mass_index == 'all':
        mass = df_mass.m.sum()
    else:
        print("mass_index should be 'all' or integer index")
    
    # integrate concentration, divide mass by concentration to get discharge
    Ct = spi.trapz(df['concentration'], dx=10.0)
    Q = (mass*1000)/Ct
            
    plt.figure()
    plt.plot(df['spec_cond'], label='spec_cond')
    plt.plot(df['background'], label='background')
    plt.xlabel('Time')
    plt.ylabel('Specific conductance')
    plt.legend(frameon=False)
    
    plt.figure()
    plt.plot(df['concentration'])
    plt.xlabel('Time')
    plt.ylabel('Concentration (mg/L)')
    plt.title('Discharge: %.3f L/s'%Q)
    
    return Q
    
