
import pandas as pd
import numpy as np

def get_event_interevent_arrays(df, key):
    """Simple method that iterates through a timeseries and looks for times when
    it is raining and when it is not.
    
    Input:
    df : Dataframe with precipitation data
    key : the column in df, assuming that time unit is the same as the index
          e.g. mm/hr and the index is in hrs. 

    Output:
    storm_depths : array of storm depths (length unit same as input)
    storm_durs : array of storm durations (time unit same as input)
    interstorm_durs : array of interstorm durations (time unit same as input)
    """

    i = 0
    interstorm_durs = []
    while i < len(df):
        Ni = 0
        while i < len(df) and df[key][i] == 0.:
            Ni += 1
            i += 1
        if Ni > 0:
            interstorm_durs.append(Ni) 
        i += 1
        
    i = 0
    storm_durs = []
    storm_depths = []
    while i < len(df):
            
        Np = 0
        depth = 0
        while i < len(df) and df[key][i] > 0:
            Np += 1
            depth += df[key][i]
            i += 1
        if Np > 0:
            storm_durs.append(Np)
            storm_depths.append(depth)
        i += 1
            
    storm_depths = np.array(storm_depths)
    storm_durs = np.array(storm_durs)
    interstorm_durs = np.array(interstorm_durs)

    return storm_depths, storm_durs, interstorm_durs

