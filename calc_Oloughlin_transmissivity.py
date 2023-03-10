# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 09:42:07 2023

@author: josep
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
# import matlab.engine
from os import listdir

def baseflow_filter(y, a, BFImax):
    """
    Implementation of the Eckhardt 2005 (HP) recursive digital filter for baseflow separation. Courtesy of Dr. Ciaran Harman. 
    """
    N = len(y) # length of the timeseries
    C = (1 - a) * BFImax / (1 - BFImax)
    A = a / (1 + C)
    B = C / (1 + C)
    b = np.zeros(N)
    b[0] = BFImax * np.mean(y)
    for k in range(1,N):
        if np.isnan(y[k]):
            b[k] = A * b[k-1]
        else:
            b[k] = A * b[k-1] + B * y[k]
        if b[k]>y[k]:
            b[k] = y[k]
    return b

def mouse_event(event):
    """
    Parameters
    ----------
    event : mouse click
       This is used so that index points can be found by clicking at any point on the graph.

    Returns
    -------
    None.

    """
    print('x: {} and y: {}'.format(event.xdata, event.ydata))
    
def hydrograph(Q, P, baseflow,titlename ='Elder Creek Hydrograph'):
    """
    Creates a hydrograph with timeseries of discharge, precipitation, and baseflow. Returns a dataframe with pertinent information to manually seperate storm events
    Parameters
    ----------
    Q : Pandas Dataframe
        Column names: 'year', 'month', 'day', 'hour', 'minute', 'second', 'discharge (mm/day)'
    P : Pandas Dataframe
        Column names: 'year', 'month', 'day', 'hour', 'minute', 'second', 'precipitation (mm/day)'
    baseflow : Arraay of float64
        timeseries of baseflow (mm/day) in 15 minute segments in one column 

    Returns
    -------
    None.

    """
    x = Q.index
    y1 = baseflow
    y2 = Q['discharge (mm/day)']
    y3 = P['precipitation (mm/day)']
    
    fig, ax1 = plt.subplots(figsize=(12,10))
    #Implementing mouse_event for the hydrograph
    cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
    fig.subplots_adjust(bottom=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x, y3, color = 'b')
    ax2.set_ylabel(('Rainfall mm/hour'),
                fontdict={'fontsize': 12})
    # Invert y axis
    ax2.invert_yaxis()
    # Primary axes
    ax1.plot(x, y1, color = 'r')#, linestyle='dashed', linewidth=1, markersize=12)
    ax1.plot(x, y2, color = 'k')#, linestyle='dashed', linewidth=1, markersize=12)
    # Define Labels
    ax1.set_xlabel(('Date'),
               fontdict={'fontsize': 14})
    
    ax1.set_ylabel(('Flow (mm/hour)'),
                   fontdict={'fontsize': 14})
    
    ax1.set_title(titlename, color = 'g')
    
    legend = fig.legend()
    
    ax1.legend(['Baseflow', 'Original Streamflow'], loc='upper left', ncol=2, bbox_to_anchor=(-.01, 1.09))
    ax2.legend(['Rainfall'], loc='upper right', ncol=1, bbox_to_anchor=(1.01, 1.09))
    
    #creating DataFrame for Manual Event Seperation Analysis with baseflow, rain, and flow
    foranalysis = pd.DataFrame({'baseflow': y1,'flow': y2,'rain': y3})
    foranalysis['flow-baseflow'] = foranalysis['flow'] - foranalysis['baseflow']
    return foranalysis


def event_sums(Q, P, baseflow, ind):
    """
    Takes event start and end indicies and cumulates total precipitation and discharge amounts for each event. 
    Parameters
    ----------
    Q : Pandas Dataframe
        Column names: 'year', 'month', 'day', 'hour', 'minute', 'second', 'discharge (mm/day)'
    P : Pandas Dataframe
        Column names: 'year', 'month', 'day', 'hour', 'minute', 'second', 'precipitation (mm/day)'
    baseflow : Arraay of float64
        timeseries of baseflow (mm/day) in 15 minute segments in one column 
    ind : Pandas Dataframe
        Columns left to right: 'event number', 'index_startflow', 'index_endflow', 'index_startrain', 'index_endrain'. All integers.  

    Returns
    -------
    Psumpts : Array of float64
        Precipitation sum (mm/day) for each event in order
    Qsumpts : Array of float64
        Discharge sum (mm/day) for each event in order

    """
    P1 = P['precipitation (mm/day)']
    Q1 = Q['discharge (mm/day)']
    Qf = Q1 - baseflow
    ind = ind.astype(int)

    #Making lists that include the sums for each event
    Psumpts = []
    for i in range(len(ind)):
        Psum = P1[ind['index_startrain'][i]:ind['index_finishrain'][i]].sum()* 0.25 # mm/15 min. find a permanent solution
        Psumpts.append(Psum)
    
    Qsumpts = []
    for i in range(len(ind)):
        Qsum = (((Qf[ind['index_startflow'][i]:ind['index_finishflow'][i]])).sum())* 0.25 # mm/15 min. find a permanent solution
        Qsumpts.append(Qsum)
    
    ### To weed out events that have a very small discharge sum
    
    # for x in range(0, len(Qsumpts)):
    #     tol = 0.2
    #     if( Qsumpts[x] < tol):
    #         Qsumpts[x] = 0  
    
    Psumpts = np.array(Psumpts)
    Qsumpts = np.array(Qsumpts)
    return Psumpts, Qsumpts

def plot_Psumpts_vs_Qsumpts(Psumpts, Qsumpts,titlename ='Elder Creek'):
    """
    
    Parameters
    ----------
    Psumpts : Array of float64
        Precipitation sum (mm/day) for each event in order
    Qsumpts : Array of float64
        Discharge sum (mm/day) for each event in order

    Returns
    -------
    None.

    """
    Pnon_zero = Psumpts>0 
    Qnon_zero = Qsumpts>0 
    
    fig, ax = plt.subplots()
    ax.scatter((Psumpts),(Qsumpts+0.1)) #Adding 0.1 to make the events with some precip values but no flow show. (so 10^-1 flow is really 0)
    ax.plot((Psumpts.min(), Psumpts.max()),(Psumpts.min(), Psumpts.max()),'k:')
    ax.set_xlim((Qsumpts[Qnon_zero].min()*0.3, Psumpts.max()*3))
    ax.set_ylim((Qsumpts[Qnon_zero].min()*0.3, Psumpts.max()*3))
    ax.set_xscale('log')
    ax.set_yscale('log') 
    ax.set_title(titlename)
    ax.set_ylabel('Cumulative Flow Sum per event')
    ax.set_xlabel("Cumulative Precip Sum per event")
    plt.show()

def Oloughlin_graph(Psumpts, Qsumpts, baseflow, titlename ='Oloughln plot'):
    """

    Parameters
    ----------
    Psumpts : Array of float64
        Precipitation sum (mm/day) for each event in order
    Qsumpts : Array of float64
        Discharge sum (mm/day) for each event in order
    baseflow : Arraay of float64
        timeseries of baseflow (mm/day) in 15 minute segments in one column 

    Returns
    -------
    OData : Pandas Dataframe
        Columns from right to left: '1/Qo' (1 over the initial baseflow for each event), 'RR_event' (runoff ratio for each event)

    """

    QPpts = pd.DataFrame()
    QPpts['Qpts'] = Qsumpts
    QPpts['Ppts'] = Psumpts
    QPpts['QPpts'] = QPpts['Qpts']/QPpts['Ppts']
    #sorts out any runoff ratios (QPpts) that are zero
    QPpts['QPpts'] = QPpts['QPpts'].fillna(0)
    QPpts = QPpts.dropna()
    
    
    #for events that have a runoff ratio that is greater than one, set to 1. 
    for i in range(0,len(QPpts)):
        if  QPpts['QPpts'][i] >= 1:
            QPpts['QPpts'][i] = 1
    
    #making a list of initial discharge values for the oloughlin plot
    Qopts = pd.DataFrame()
    Qopts['startflowindex'] = ind['index_startflow']
    Qopts = Qopts.dropna()
                                  
    Qo = []
    for i in range(0,len(Qopts)):
        Qo.append(baseflow[int(Qopts['startflowindex'][i])-1]) #BASEFLOW instead of flow
    Qo = pd.DataFrame(Qo)
    Qo['1.Qo'] = (1/Qo[0])
    
    #putting all this data together
    figdata = pd.DataFrame()
    figdata['Qo'] = Qo[0]
    figdata['1/Qo'] = Qo['1.Qo']
    figdata['Q/P Points'] = QPpts['QPpts']
    figdata['Precip sums'] = QPpts['Ppts']
    figdata['Q sums'] = QPpts['Qpts']
    
    ### To implement for different tolerances for the events
    
    # for i in range(0,len(figdata)):
    #     # if QPpts['Ppts'][i] < 0.1:
    #     #     figdata = figdata.drop(i)
    #     if QPpts['QPpts'][i] >= 1:
    #         figdata = figdata.drop(i)
    #     # elif QPpts['Ppts'][i] > 60:
    #     #     figdata = figdata.drop(i)
    
    #fitting the a curve to the storm events
    Qo = np.array(figdata['Qo'])
    R = np.array((figdata['Q/P Points'])*100)
    Q2 = Qo @ np.transpose(Qo)
    R2 = Qo @ np.transpose(R)
    theta = (1/Q2) * (R2)
    hyperbolictest = figdata['1/Qo']
    hyperbolictest = hyperbolictest.sort_values() 

    #plotting the O'Loughlin graph
    fig, axis = plt.subplots()
    axis.plot(hyperbolictest, theta / hyperbolictest)
    axis.scatter(figdata['1/Qo'], (figdata['Q/P Points'])*100)#, c=figdata['Precip sums'], s=50, cmap='Blues', norm = colors.LogNorm(), alpha = 0.5)
    #axis.text(1,80,f"theta= {theta:.3f}")
    axis.set_title(titlename)
    axis.set_xlabel('1/Qo')
    axis.set_ylabel("Q/P")
    axis.set_ylim(-5,105)
    #axis.set_xlim(0.02,100)
    axis.set_xscale('log')
    plt.show()
    
    #configuring data for other steps
    OData = pd.DataFrame()
    OData['1/Qo'] = figdata['1/Qo'] 
    OData['RR_event'] = (figdata['Q/P Points'])

    #filtering data for runoff events that are not infinity, greater than 1, or less than tolerance 0.0002
    OData = OData[OData['RR_event'] >= 0.0002]
    OData = OData[OData['RR_event'] != (-math.inf)]
    OData = OData[OData['RR_event'] != 1]
    OData = OData.reset_index()

    return OData

def transmissivity_calc(ti, OData, titlename ='Elder Creek'):
    """

    Parameters
    ----------
    ti : Pandas Dataframe
        Columns left to right: 'TI' (ti values sorted from highest to lowest, float), 'CDF_Val' (values starting at 1 on the 0 index TI value and going down to 0 on the last value)
    OData : Pandas Dataframe
        Columns from right to left: '1/Qo' (1 over the initial baseflow for each event), 'RR_event' (runoff ratio for each event)

    Returns
    -------
    T : float64
        Transmissivity Estimate
    df : Pandas Dataframe
        Columns left to right:
            'TI' (ti values sorted from highest to lowest, float)
            'CDF_Val' (values starting at 1 on the 0 index TI value and going down to 0 on the last
            '1/Qo' (log of the baseflow values before the discharge response for each event, nans to fill the rest of the column)
            '1/Qo not log' (same as 1/Qo but not logged)
            'RR_event' (runoff ratio for each event)
    b : float
        y intercept of log(Qo) vs log(TI) graph with fixed slope of 1
    a1 : float64
        Error on b. 

    """
    finaldata = ti.copy(deep=True)
    finaldata['1/Qo'] = np.log(OData['1/Qo']*1000)
    finaldata['1/Qo not log'] = (OData['1/Qo']*1000)
    finaldata['RR_event'] = OData['RR_event']
    
    #finding closest TI value in complement of the CDF to runoff ratio for each storm event
    # find neighbor code courtesy of https://stackoverflow.com/questions/30112202/how-do-i-find-the-closest-values-in-a-pandas-series-to-an-input-number
    df = finaldata
    logTIvals = []
    logQ = df['1/Qo'].dropna()
    df1 = pd.DataFrame()
    df1['logQ'] = logQ

    for j in range(0,len(logQ)):
        value = df['RR_event'][j]
        colname = 'CDF_Val'
        exactmatch = df[df[colname] == value]
        if value == 0:
            df1 = df1.drop([j])
            #logTIvals.append(4759521389)
        if value == 1:
            df1 = df1.drop([j])
        try:
            if not exactmatch.empty:
                logTIvals.append((df['TI'][(exactmatch.index).int])) 
            else:
                lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
                upperneighbour_ind = df[df[colname] > value][colname].idxmin()
                logTIvals.append(df['TI'][(lowerneighbour_ind)])
        except:
            pass
    
    #plotting logQo and logTI values with a regression with fixed slope of 1. 
    df1['logTIvals'] = np.log(logTIvals)
    df1 = df1.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    logQ = df1['logQ']
    logTIvals = df1['logTIvals']

    #finding error bounds on the intercept value (b) within 95 percent
    b = (sum(logTIvals)/len(logTIvals)) - (sum(logQ)/len(logQ))
    a = 0
    for k in logTIvals.index:
        a += (logTIvals[k] - logQ[k] - b)**2
    a1 = a / (len(logTIvals)-1)
    a1 = np.sqrt(a1)
    a1 = a1 * 1.96

    fig, ax = plt.subplots()
    ax.scatter(logQ,logTIvals,c=df1.index, s=50, cmap='Blues', alpha = 0.5)
    ax.plot(logQ, logQ + b, label='regression line') 
    ax.text(10,5,f"b= {b:.3f}")
    ax.set_title(titlename)
    ax.set_xlim(0,15)
    ax.set_ylim(0,15)
    ax.set_xlabel('log(1/Qo)')
    ax.set_ylabel('log(TI), log(A/WS)')
    plt.show()
    
    #Transmissivity Value Estimation
    T = np.exp(b)
    
    return T, df, b, a1

def final_graph(T, df, b, a1, titlename ='Elder Creek'):
    """

    Parameters
    ----------
    T : float64
        Transmissivity Estimate
    df : Pandas Dataframe
        Columns left to right:
            'TI' (ti values sorted from highest to lowest, float)
            'CDF_Val' (values starting at 1 on the 0 index TI value and going down to 0 on the last
            '1/Qo' (log of the baseflow values before the discharge response for each event, nans to fill the rest of the column)
            '1/Qo' not log (same as 1/Qo but not logged)
            'RR_event' (runoff ratio for each event)
    b : float
        y intercept of log(Qo) vs log(TI) graph with fixed slope of 1
    a1 : float64
        Error on b. 

    Returns
    -------
    None.

    """
    #T/Qo as x axis vs Runoff ratio on the y axis. (1-CDF of TI values shown as a line) 
    df2 = df.copy(deep = True)
    df2['T/Qo'] = df2['1/Qo not log']*T
    df2['logT/Qo'] = np.log(df2['T/Qo'])
    df2['logTI'] = np.log(df2['TI'])

    trans = f'{T: .2f}'
    errup = f'{(np.exp(a1+b)): .2f}'
    errdown = f'{(np.exp(b-a1)): .2f}'

    plt.figure()
    plt.scatter(df2['logT/Qo'],df2['RR_event'],c=df2.index, s=50, cmap='Blues', alpha = 1)
    plt.plot(np.log(df2['TI']),df2['CDF_Val'], color = '#1f77b4')
    plt.title(titlename)
    plt.xlabel(str("log(T/Qo) where T = " + str(trans) + " bounds(" + str(errup) + ' - ' + str(errdown) + ')'))
    plt.ylabel('Quickflow / net rainfall (%)')
    plt.text(6,0.85,"Blue Line = \nCompliment of the \nTopographic Index CDF")
    plt.xlim(1,9)
    plt.ylim(0,1.01)
    plt.show()

def run_total(Q, P, ind, ti):
    """
    Parameters
    ----------
    Q : Pandas Dataframe
        Column names: 'year', 'month', 'day', 'hour', 'minute', 'second', 'discharge (mm/day)'
    P : Pandas Dataframe
        Column names: 'year', 'month', 'day', 'hour', 'minute', 'second', 'precipitation (mm/day)'
    ind : Pandas Dataframe
        Events, columns left to right: 'event number', 'index_startflow', 'index_endflow', 'index_startrain', 'index_endrain'. All integers.  
    ti : Pandas Dataframe
        Columns left to right: 'TI' (ti values sorted from highest to lowest, float), 'CDF_val' (values starting at 1 on the 0 index TI value and going down to 0 on the last value)

    Returns
    -------
    final O'loughlin graph
    """
    baseflow = baseflow_filter(y = Q['discharge (mm/day)'],a = 0.9999, BFImax = 0.8)
    hydrograph(Q,P,baseflow)
    Psumpts, Qsumpts = event_sums(Q, P, baseflow, ind)
    plot_Psumpts_vs_Qsumpts(Psumpts, Qsumpts)
    OData = Oloughlin_graph(Psumpts, Qsumpts, baseflow)
    T, df, b, a1 = transmissivity_calc(ti,OData)
    final_graph(T, df, b, a1)

if __name__ == "__main__":
    Q = pd.read_csv("./Oloughlin_test/ElderCreek_Q_mmday.txt", sep = (' '), header = None)
    Q = Q.rename(columns={0: "year",1:"month", 2:"day", 3:"hour",4:"minute", 5:"second", 6:"discharge (mm/day)"})
    P = pd.read_csv("./Oloughlin_test/ElderCreek_P_mmday.txt", sep = (' '), header = None)
    P = P.rename(columns={0: "year",1:"month", 2:"day", 3:"hour",4:"minute", 5:"second", 6:"precipitation (mm/day)"})
    ti = pd.read_csv("./Oloughlin_test/ElderCreek_TI_CDF.csv")
    ti = ti.rename(columns={'0-1': "CDF_Val"})
    ind = pd.read_csv("./Oloughlin_test/Indices.txt", sep = ('\t'))
    
    run_total(Q, P, ind, ti)

