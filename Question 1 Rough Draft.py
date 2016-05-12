
# coding: utf-8

# In[1]:

from urllib import request
import matplotlib
from datetime import datetime
from pytz import timezone
matplotlib.rcParams['figure.figsize'] = [12.0, 9.0]
import numpy as np
import pyart
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import urllib.request
with urllib.request.urlopen('http://www.atmos.anl.gov/ANLMET/format.txt') as f:
    print(f.read(300)) 


# In[2]:

with urllib.request.urlopen('http://www.atmos.anl.gov/ANLMET/format.txt') as f:
    print(f.read(100).decode('utf-8'))


# In[3]:

fig, ax1 = plt.subplots()
t = np.arange(0.01, 10.0, 0.01)
s1 = np.exp(t)
ax1.plot(t, s1, 'b-')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('drybulb', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')


ax2 = ax1.twinx()
s2 = np.sin(2*np.pi*t)
ax2.plot(t, s2, 'r-')
ax2.set_ylabel('dew point', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.show()


# In[4]:

date_str = "2016-05-28 22:28:15"
datetime_obj_naive = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")


# In[5]:

from datetime import datetime
from pytz import timezone
fmt = "%Y-%m-%d %H:%M:%S %Z%z"

now_utc = datetime.now(timezone('UTC'))
print(now_utc.strftime(fmt))

now_pacific = now_utc.astimezone(timezone('US/Pacific'))
print(now_pacific.strftime(fmt))
now_berlin = now_pacific.astimezone(timezone('Europe/Berlin'))
print(now_berlin.strftime(fmt))


# In[6]:

from pytz import all_timezones

print(len(all_timezones))
for zone in all_timezones:
    if 'US' in zone:
        print(zone)


# In[11]:

fig, ax1 = plt.subplots()
t = np.arange(0.01, 10.0, 0.01)
s1 = np.exp(t)
ax1.plot(t, s1, 'b-')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('time', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
s2 = np.sin(2*np.pi*t)
ax2.plot(t, s2, 'r.')
ax2.set_ylabel('date', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.show()


# In[69]:

fig, ax1 = plt.subplots()
t = np.arange(0.01, 10.0, 0.01)
s1 = np.exp(t)
ax1.plot(t, s1, 'b-')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('time', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')
    
ax1.plot(r.date, r.close, lw=2)
ax2.fill_between(r.date, pricemin, r.close, facecolor='blue', alpha=0.5))    

ax2 = ax1.twinx()
s2 = np.sin(2*np.pi*t)
ax2.plot(t, s2, 'r.')
ax2.set_ylabel('date', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.show()


# In[87]:

fig, ax1 = plt.subplots()
t = np.arange(0.01, 10.0, 0.01)
s1 = np.exp(t)
ax1.plot(t, s1, 'b-')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('time', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
s2 = np.sin(2*np.pi*t)
ax2.plot(t, s2, 'r.')
ax2.set_ylabel('date', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.show()

plt.fill_between(t, s2, color='green', alpha='1')
plt.show()


# In[91]:

import numpy as np
csv_data = 'http://www.atmos.anl.gov/ANLMET/format.txt'


# In[92]:

plt.plot((17.14, 17.86, 18.67, 19.37, 19.94, 20.51, 20.46, 20.45, 20.68, 21.45, 22.10, 22.49, 22.60, 22.91, 23.40, 23.58, 24.03, 24.28, 24.71,
         ))


# In[93]:

datetime.datetime.strptime('16Jan2016', '%d%b%Y')
datetime.datetime(2016, 1, 16)


# In[2]:

import matplotlib as mpl

norm = mpl.colors.Normalize(vmin=-1.,vmax=1.)

norm(0.)
Out[3]: 0.5


# In[3]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal

N = 100
X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
Z1 = (bivariate_normal(X, Y, 1., 1., 1.0, 1.0))**2      - 0.4 * (bivariate_normal(X, Y, 1.0, 1.0, -1.0, 0.0))**2
Z1 = Z1/0.03



fig, ax = plt.subplots(3, 1, figsize=(8, 8))
ax = ax.flatten()
bounds = np.linspace(-1, 1, 10)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
pcm = ax[0].pcolormesh(X, Y, Z1,
                       norm=norm,
                       cmap='RdBu_r')
fig.colorbar(pcm, ax=ax[0], extend='both', orientation='vertical')

bounds = np.array([-0.25, -0.125, 0, 0.5, 1])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
pcm = ax[1].pcolormesh(X, Y, Z1, norm=norm, cmap='RdBu_r')
fig.colorbar(pcm, ax=ax[1], extend='both', orientation='vertical')

pcm = ax[2].pcolormesh(X, Y, Z1, cmap='RdBu_r', vmin=-np.max(Z1))
fig.colorbar(pcm, ax=ax[2], extend='both', orientation='vertical')
fig.show()


# In[2]:

from urllib import request
import matplotlib
from datetime import datetime
from pytz import timezone
matplotlib.rcParams['figure.figsize'] = [12.0, 9.0]
import numpy as np
import pyart
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import urllib.request
with urllib.request.urlopen('http://www.atmos.anl.gov/ANLMET/format.txt') as f:
    print(f.read(300))  


# In[3]:

date_str = "2016-05-05 22:28:15"
datetime_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
datetime_obj_utc = datetime_obj.replace(tzinfo=timezone('UTC'))
print(datetime_obj_utc.strftime("%Y-%m-%d %H:%M:%S %Z%z"))


# In[4]:

date_str = "2016-05-28 22:28:15"
datetime_obj_naive = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
from datetime import datetime
from pytz import timezone
fmt = "%Y-%m-%d %H:%M:%S %Z%z"

now_utc = datetime.now(timezone('UTC'))
print(now_utc.strftime(fmt))

now_pacific = now_utc.astimezone(timezone('US/Pacific'))
print(now_pacific.strftime(fmt))
now_berlin = now_pacific.astimezone(timezone('Europe/Berlin'))
print(now_berlin.strftime(fmt))


# In[5]:

import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
import pyart.graph
import tempfile
import pyart.io
import boto


# In[6]:

radar.azimuth.keys()


# In[7]:

radar.azimuth['data']


# In[8]:

fin - open(NEXRAD_FILE, 'rb')
out = open(OUTPUT_FILE, 'wb')


# In[9]:

out.write(fin.read(24 + 12 + RECORD_SIZE * 134))


# In[10]:

import numpy as np
csv_data = 'http://www.atmos.anl.gov/ANLMET/format.txt'
#### %matplotlib inlxine
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.plot((17.14, 17.86, 18.67, 19.37, 19.94, 20.51, 20.46, 20.45, 20.68, 21.45, 22.10, 22.49, 22.60, 22.91, 23.40, 23.58, 24.03, 24.28, 24.71,
         ))


# In[11]:

etime.datetime.strptime('16Jan2016', '%d%b%Y')
datetime.datetime(2016, 1, 16)


# In[12]:

datetime.datetime.strptime('16Jan2016', '%d%b%Y')
datetime.datetime(2016, 1, 17)


# In[13]:

datetime.datetime.strptime('16Jan2016', '%d%b%Y')
datetime.datetime(2016, 1, 18)


# In[14]:

import matplotlib.pyplot as plt
import datetime
import numpy as np


# In[16]:

x = np.array([datetime.datetime(2016, 1, 16, i, 0) for i in range(24)])
y = np.random.randint(31, size=x.shape)
plt.plot(x,y)
plt.show()


# In[17]:

y2 = np.ma.masked_greater(y2, 1.0)
ax1.plot(x, y1, x, y2, color='black')
ax1.fill_between(x, y1, y2, where=y2 >= y1, facecolor='green', interpolate=True)
ax1.fill_between(x, y1, y2, where=y2 <= y1, facecolor='red', interpolate=True)
ax1.set_title


# In[18]:


t = np.arange(0.01, 10.0, 0.01)
s1 = np.exp(t)
ax1.plot(t, s1, 'b-')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('time', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')
ax2 = ax1.twinx()
s2 = np.sin(2*np.pi*t)
ax2.plot(t, s2, 'r.')
ax2.set_ylabel('date', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.show()

ax3.fill_between(x, y1, y2)
ax3.set_ylabel('between x1 and x2')
ax3.set_xlabel('x')
fig, ax1 = plt.subplots()
fig, (ax, ax1) = plt.subplots(2, 1, sharex=True)
ax.plot(x, y1, x, y2, color='green')
ax1.fill_between(x, y1, y2, where=y2 >= y1, facecolor='green', interpolate=True)
ax2.fill_between(x, y1, y2, where=y2 <= y1, facecolor='green', interpolate=True)
ax.set_title


# In[19]:

from pylab import *
from matplotlib import rc, rcParams

rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})



dataMatrix1 = genfromtxt('http://www.atmos.anl.gov/ANLMET/format.txt')


x = dataMatrix1[:,0]
y1 = dataMatrix1[:,1]
y2 = dataMatrix1[:,2]

plot(x,y1,label=r'f(x) = x^2')
plot(x,y2,label=r'f(x) = x^3')

legend(loc='upper right')

xlabel(r'$x$',fontsize=16)
plt.ylabel(r'$f(x)$',fontsize=16)

show()


# In[ ]:



