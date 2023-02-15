import numpy as np
import pandas as pd
import glob
import rasterio as rd
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

save_directory = 'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures/'

# %% paths


## Soldiers Delight:
path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/'
slopefile = "LSDTT/baltimore2015_DR1_SLOPE.bil"
areafile = "LSDTT/baltimore2015_DR1_d8_area.bil"

## Oregon Ridge
# path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/'
# slopefile = "LSDTT/baltimore2015_BR_SLOPE.bil"
# areafile = "LSDTT/baltimore2015_BR_d8_area.bil"

#%% import for topographic index

# fitted (10 m) slope
sf = rd.open(path+slopefile)
slope = sf.read(1).astype(float)
slope = np.ma.masked_array(slope, mask=slope==-9999)

dx = sf.transform[0]
bounds = sf.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
# sf.close()

# d8 area
af = rd.open(path+areafile)
area = af.read(1).astype(float)
area = np.ma.masked_array(area, mask=area==-9999)
af.close()

#%%
# calculate and filter topographic index 

TI = np.log(area/(slope * dx))

plt.figure()
plt.imshow(TI,
            origin="upper", 
            extent=Extent,
            cmap='viridis',
            )
plt.show()

TI_filtered = gaussian_filter(TI, sigma=4)

plt.figure()
plt.imshow(TI_filtered,
            origin="upper", 
            extent=Extent,
            cmap='viridis',
            )
plt.show()

# write TI filtered to .tif
TIfile = "LSDTT/baltimore2015_DR1_TIfiltered.tif" # Druids Run
# TIfile = "LSDTT/baltimore2015_BR_TIfiltered.tif" # Baisman Run
sf = rd.open(path+slopefile)
TI_dataset = rd.open(
    path+TIfile,
    'w',
    driver='GTiff',
    height=sf.height,
    width=sf.width,
    count=1,
    dtype=TI_filtered.dtype,
    crs=sf.crs,
    transform=sf.transform,
)
TI_dataset.write(TI_filtered,1)
TI_dataset.close()

#%% clean Pond Branch and Soldiers Delight EMLID REACH saturation files

path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/saturation/'
paths = glob.glob(path + "PB_UTM_*.csv")

for pth in paths:
    df = pd.read_csv(pth)
    df = df[['Name', 'Easting', 'Northing', 'Averaging start', 'Easting RMS', 'Northing RMS']]
    A = [True if X in ['N', 'Ys', 'Yp', 'Yf'] else False for X in df['Name']]
    df = df[A]

    namesdict = {'Easting':'X',
                'Northing': 'Y',
                'Averaging start': 'BeginTime',
                'Easting RMS': 'X_rms', 
                'Northing RMS': 'Y_rms',
                }
    df.rename(columns=namesdict, inplace=True)
    df.to_csv(path + "transects_%s.csv"%pth.split('_')[-1][:-4])


path = 'C:/Users/dgbli/Documents/Research/Soldiers Delight/data/saturation/'
paths = glob.glob(path + "DR_UTM_*.csv")

for pth in paths:
    df = pd.read_csv(pth)
    df = df[['Name', 'Easting', 'Northing', 'Averaging start', 'Easting RMS', 'Northing RMS']]
    A = [True if X in ['N', 'Ys', 'Yp', 'Yf'] else False for X in df['Name']]
    df = df[A]

    namesdict = {'Easting':'X',
                'Northing': 'Y',
                'Averaging start': 'BeginTime',
                'Easting RMS': 'X_rms', 
                'Northing RMS': 'Y_rms',
                }
    df.rename(columns=namesdict, inplace=True)
    df.to_csv(path + "transects_%s.csv"%pth.split('_')[-1][:-4])
