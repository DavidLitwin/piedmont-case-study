"""
Borrowing extensively from https://here.isnew.info/how-to-save-a-numpy-array-as-a-geotiff-file-using-gdal.html

"""
#%%
import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from landlab.io.netcdf import from_netcdf

def read_envi(filename):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, ds

def write_envi(filename, arr, dx, in_ds):

    driver = gdal.GetDriverByName("ENVI")
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    out_ds.SetProjection(in_ds.GetProjection())

    in_gt = in_ds.GetGeoTransform()
    out_gt = (in_gt[0], dx, -0.0, in_gt[3], -0.0, -dx)
    out_ds.SetGeoTransform(out_gt)

    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)


#%% Read some existing files to get geotransform

# path = 'C:/Users/dgbli/Documents/Research/Oregon Ridge/data/LSDTT/'
path = '/Users/dlitwin/Documents/Research/Oregon Ridge/data/LSDTT/'
name = 'baltimore2015_BR_hs.bil' # hillshade

bais_arr, bais_ds = read_envi(path+name)

#%% load landlab grids and export


# directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc/'
directory = '/Users/dlitwin/Documents/Research Data/HPC output/DupuitLEMResults/post_proc/'
base_output_path = 'CaseStudy_cross_1'
model_runs = range(4)
# fields = {'topographic__elevation':'', 'qstar':'qstar', 'curvature':'curv'}
fields = {'topographic__elevation':'', 'curvature':'curv'}

for i in model_runs:
    grid = from_netcdf(os.path.join(directory,base_output_path, f'grid_{i}.nc')) 

    for key, val in fields.items():
            
        arr = grid.at_node[key].reshape(grid.shape)
        arr_0 = arr[1:-1, 1:-1]
        arr_1 = np.pad(arr_0, ((5,5),(0,5)), mode='symmetric')
        arr_2 = np.flipud(arr_1)
        
        if val == '':
            write_envi(os.path.join(directory,base_output_path, 'lsdtt', f'{base_output_path}-{i}.bil'), 
                    arr_2, grid.dx, bais_ds)
        else:
            write_envi(os.path.join(directory,base_output_path, 'lsdtt', f'{base_output_path}-{i}_{val}.bil'), 
            arr_2, grid.dx, bais_ds)


# %%
