
# coding: utf-8

# Its technical name is WSR-88D, which stands for Weather Surveillance Radar, 1988, Doppler. NEXRAD detects precipitation and atmospheric movement or wind. It returns data which when processed can be displayed in a mosaic map which shows patterns of precipitation and its movement.

# In[37]:

import pyart
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
print(pyart.__version__)


# In[84]:

filename = './Downloads/KLOT20130417_235520_V06.gz'
radar = pyart.io.read('./Downloads/KLOT20130417_235520_V06.gz')


# In[39]:

dir(radar)


# In[6]:

pyart.io.write_cfradial('./Downloads/KLOT20130417_235520_V06.gz', radar)


# In[40]:

radar.azimuth.keys()


# In[41]:

radar.azimuth['standard_name']


# In[42]:

radar.azimuth['data']


# In[43]:

f = plt.figure(figsize=[15,8])
plt.plot(radar.time['data'], radar.azimuth['data'] )
plt.xlabel(radar.time['standard_name'] + ' (' + radar.time['units'] + ')')
plt.ylabel(radar.azimuth['standard_name'] + ' (' + radar.azimuth['units'] + ')')


# In[45]:

print(radar.range['data'].min(), radar.range['data'].max(), radar.range['units'])
f = plt.figure(figsize=[15,8])
plt.plot(radar.time['data'], radar.elevation['data'] )
plt.xlabel(radar.time['standard_name'] + ' (' + radar.time['units'] + ')')
plt.ylabel(radar.elevation['standard_name'] + ' (' + radar.elevation['units'] + ')')


# In[47]:

for mykey in radar.metadata.keys():
    print(mykey, ': ', radar.metadata[mykey])


# In[48]:

radar.sweep_end_ray_index['data']
f = plt.figure(figsize=[15,8])
for i in range(len(radar.sweep_end_ray_index['data'])):
    start_index = radar.sweep_start_ray_index['data'][i]
    end_index = radar.sweep_end_ray_index['data'][i]
    plt.plot(radar.time['data'][start_index:end_index], 
             radar.elevation['data'][start_index:end_index], 
             label = 'Sweep number '+ str(radar.sweep_number['data'][i]))
plt.legend()
plt.xlabel(radar.time['standard_name'] + ' (' + radar.time['units'] + ')')
plt.ylabel(radar.elevation['standard_name'] + ' (' + radar.elevation['units'] + ')')


# In[50]:

print(radar.fields.keys())
print("")
for mykey in radar.fields.keys():
    print(mykey,':', radar.fields[mykey]['standard_name'] + ' (' + radar.fields[mykey]['units'] + ')')


# In[73]:

print(radar.fields.keys())


# In[77]:

radar = radar.extract_sweeps([0,1])


# In[79]:

display = pyart.graph.RadarMapDisplay(radar)
f = plt.figure(figsize = [17,4])
plt.subplot(1, 3, 1) 


# In[80]:

display = pyart.graph.RadarDisplay(radar)
display.plot('reflectivity', vmin=-16, vmax=80, cmap='pyart_NWSRef')


# In[3]:

radar.info('compact')  


# In[1]:

display.plot_ppi_map('reflectivity_horizontal', 1, vmin=-20, vmax=20,
                     min_lon=-157.1, max_lon=-156, min_lat=71.2, max_lat=71.6,
                     lon_lines=np.arange(-158, -154, .2), projection='lcc',
                     lat_lines=np.arange(69, 72, .1), resolution='h',
                     lat_0=radar.latitude['data'][0],
                     lon_0=radar.longitude['data'][0])


# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import pyart
filename = './Downloads/KLOT20130417_235520_V06.gz'
radar = pyart.io.read('./Downloads/KLOT20130417_235520_V06.gz')
display = pyart.graph.RadarMapDisplay(radar)


display.plot_ppi_map('reflectivity_horizontal', 1, vmin=-20, vmax=20,
                     min_lon=-157.1, max_lon=-156, min_lat=71.2, max_lat=71.6,
                     lon_lines=np.arange(-158, -154, .2), projection='lcc',
                     lat_lines=np.arange(69, 72, .1), resolution='h',
                     lat_0=radar.latitude['data'][0],
                     lon_0=radar.longitude['data'][0])


display.plot_range_ring(10., line_style='k-')
display.plot_range_ring(20., line_style='k--')
display.plot_range_ring(30., line_style='k-')
display.plot_range_ring(40., line_style='k--')


display.plot_line_xy(np.array([-40000.0, 40000.0]), np.array([0.0, 0.0]),
                     line_style='k-')
display.plot_line_xy(np.array([0.0, 0.0]), np.array([-20000.0, 200000.0]),
                     line_style='k-')


display.plot_point(radar.longitude['data'][0], radar.latitude['data'][0])

plt.show()


# In[ ]:

import pyart
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import urllib.request
import numpy as np
print(pyart.__version__)
with urllib.request.urlopen('http://www.ndbc.noaa.gov/measdes.shtml#srad') as f:
    print(f.read(300))
print(radar.fields.keys())   
pyart.io.write_cfradial('converted_sigmet_file.nc', radar) 
display = pyart.graph.RadarMapDisplay(radar)
f = plt.figure(figsize = [17,4])
plt.subplot(1, 3, 1)
isplay = pyart.graph.RadarMapDisplay(radar)
radar = radar.extract_sweeps([0, 1])
display = pyart.graph.RadarMapDisplay(radar)
f = plt.figure(figsize = [17,4])
plt.subplot(1, 3, 1) 
display = pyart.graph.RadarMapDisplay(radar)
f = plt.figure(figsize = [17,4])
plt.subplot(1, 3, 1) 
display.plot_ppi_map('differential_reflectivity', max_lat = 26.5, min_lat =25.4, min_lon = -81., max_lon = -79.5,
                     vmin = -7, vmax = 7, lat_lines = np.arange(20,28,.2), lon_lines = np.arange(-82, -79, .5),
                     resolution = 'i')
plt.subplot(1, 3, 2) 
display.plot_ppi_map('reflectivity', max_lat = 26.5, min_lat =25.4, min_lon = -81., max_lon = -79.5,
                     vmin = -8, vmax = 64, lat_lines = np.arange(20,28,.2), lon_lines = np.arange(-82, -79, .5),
                     resolution = 'i')
plt.subplot(1, 3, 3) 
display.plot_ppi_map('velocity', sweep = 1, max_lat = 26.5, min_lat =25.4, min_lon = -81., max_lon = -79.5,
                     vmin = -15, vmax = 15, lat_lines = np.arange(20,28,.2), lon_lines = np.arange(-82, -79, .5),
                     resolution = 'i')
display = pyart.graph.RadarDisplay(radar)
fig = plt.figure(figsize=(9, 12))

plots = [
  
    ['reflectivity', 'Reflectivity (dBZ)', 0],
    ['differential_reflectivity', 'Zdr (dB)', 0],
    ['differential_phase', 'Phi_DP (deg)', 0],
    ['cross_correlation_ratio', 'Rho_HV', 0],
    ['velocity', 'Velocity (m/s)', 1],
    ['spectrum_width', 'Spectrum Width', 1]
]

def plot_radar_images(plots):
    ncols = 2
    nrows = len(plots)/2
    for plotno, plot in enumerate(plots, start=1):
        ax = fig.add_subplot(nrows, ncols, plotno)
        display.plot(plot[0], plot[2], ax=ax, title=plot[1],
             colorbar_label='',
             axislabels=('East-West distance from radar (km)' if plotno == 6 else '', 
                         'North-South distance from radar (km)' if plotno == 1 else ''))
        display.set_limits((-300, 300), (-300, 300), ax=ax)
        display.set_aspect_ratio('equal', ax=ax)
        display.plot_range_rings(range(100, 350, 100), lw=0.5, col='black', ax=ax)
    plt.show()

plot_radar_images(plots)
radar.info('compact')


# In[4]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal

'''

'''
N = 100
X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]

Z1 = bivariate_normal(X, Y, 0.1, 0.2, 1.0, 1.0) +      0.1 * bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)

fig, ax = plt.subplots(2, 1)

pcm = ax[0].pcolor(X, Y, Z1,
                   norm=colors.LogNorm(vmin=Z1.min(), vmax=Z1.max()),
                   cmap='PuBu_r')
fig.colorbar(pcm, ax=ax[0], extend='max')

pcm = ax[1].pcolor(X, Y, Z1, cmap='PuBu_r')
fig.colorbar(pcm, ax=ax[1], extend='max')
fig.show()


# In[5]:

nexrad_site = 'klot'
f = plt.figure(figsize = [24,9])
display = pyart.graph.RadarMapDisplay(radar)
display.plot_ppi_map(
    'reflectivity', vmin=-32, vmax=80, cmap='pyart_NWSRef',
    resolution='i', embelish=False)
display.basemap.drawcounties()
display.plot_crosshairs(lon=-87.5987, lat=41.7886)


# In[ ]:



