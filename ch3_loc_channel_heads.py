"""
Use matplotlib to pick out channel heads from DupuitLEM results 

Needs backend: %matplotlib qt6

"""

#%%
import os
import numpy as np
import pandas as pd

import rasterio as rd
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_point_clicker import clicker

# directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
directory = '/Users/dlitwin/Documents/Research Data/HPC output/DupuitLEMResults/post_proc/'
base_output_path = 'CaseStudy_cross_15'
i = 2

#%% Hillshades (projected coordinates) - define channel network

# path = directory + f'/{base_output_path}/'
# name_elev = '%s-%d_pad.bil'%(base_output_path, i) # elevation
# src_elev = rd.open(path + name_elev) # elevation
# name_elev = '%s-%d.bil'%(base_output_path, i) # elevation
# path = os.path.join(directory,base_output_path,'lsdtt', name_elev)
# path = os.path.join(directory,base_output_path, name_elev)
# src_elev = rd.open(path) # elevation

src_elev = rd.open(os.path.join(directory, base_output_path, 'lsdtt', '%s-%d.bil'%(base_output_path, i)))
src_curv = rd.open(os.path.join(directory, base_output_path, 'lsdtt', '%s-%d_curv.bil'%(base_output_path, i)))
curv = src_curv.read(1)
channels = 1.0 * (curv > 0.001)

bounds = src_elev.bounds
Extent = [bounds.left,bounds.right,bounds.bottom,bounds.top]
proj = src_elev.crs
utm = 18

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.UTM(utm))
ls = LightSource(azdeg=135, altdeg=45)
klicker = clicker(ax, ["source"], markers=["o"])
ax.set_extent(Extent, crs=ccrs.UTM(utm))
cs = ax.imshow(
                ls.hillshade(src_elev.read(1), 
                             vert_exag=1),
                cmap='gray', 
                #vmin=100,
                extent=Extent, 
                transform=ccrs.UTM(utm), 
                origin="upper")
im = ax.imshow(np.flipud(channels), 
                origin="lower", 
                extent=Extent,
                # extent=(x[0], x[-1], y[0], y[-1]), 
                cmap='Reds',
                alpha=np.flipud(channels),
                vmin=0.0,
                vmax=1,
                interpolation=None,
                )
plt.show()

#%% get points, save output

out = klicker.get_positions()
pts_utm = out['source']
geo = ccrs.Geodetic()
pts_latlon = geo.transform_points(ccrs.UTM(utm), pts_utm[:,0], pts_utm[:,1])

df_sources = pd.DataFrame({'x':pts_utm[:,0], 
                        'y':pts_utm[:,1], 
                        'latitude':pts_latlon[:,1], 
                        'longitude':pts_latlon[:,0]})
df_sources.to_csv('%s/%s/%s-%d_EyeSources_1.csv'%(directory, base_output_path, base_output_path, i))

# %% hillshades + curvature delineation

df = pd.read_csv('%s/%s/%s-%d_EyeSources_1.csv'%(directory, base_output_path, base_output_path, i))
# %%
