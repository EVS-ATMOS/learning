
# coding: utf-8

# In[40]:

from urllib import request


# In[41]:

anl_url = 'http://www.atmos.anl.gov/ANLMET/anltower.48'


# In[42]:

def download_stock_data(csv_url):
    response =  request.urlopen(csv_url)
    csv = response.read()
    csv_str = str(csv)
    lines = csv_str.split('\\n')
    dest_url = 'anl.csv'
    fx = open(dest_url, 'w')
    for line in lines:
        fx.write(line + '\n')
    fx.close()
    
download_stock_data(anl_url)


# In[64]:

numbers = raw_input('TaC_60m')
list_of_numbers = number.split()
numbersInt = map(int, list_of_numbers)
min(numbersInt)
max(numbersInt)


# In[44]:

import numpy as np
csv_data = 'http://www.atmos.anl.gov/ANLMET/anltower.48'


# In[45]:

#### %matplotlib inlxine
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[46]:

plt.plot((17.14, 17.86, 18.67, 19.37, 19.94, 20.51, 20.46, 20.45, 20.68, 21.45, 22.10, 22.49, 22.60, 22.91, 23.40, 23.58, 24.03, 24.28, 24.71,
         ))


# In[60]:

datetime.datetime.strptime('16Jan2016', '%d%b%Y')
datetime.datetime(2016, 1, 16)

    



# In[59]:

datetime.datetime.strptime('16Jan2016', '%d%b%Y')
datetime.datetime(2016, 1, 17)


# In[61]:

datetime.datetime.strptime('16Jan2016', '%d%b%Y')
datetime.datetime(2016, 1, 18)


# In[66]:

import matplotlib.pyplot as plt
import datetime
import numpy as np


# In[67]:

x = np.array([datetime.datetime(2016, 1, 16, i, 0) for i in range(24)])
y = np.random.randint(31, size=x.shape)


# In[68]:

plt.plot(x,y)
plt.show()


# In[ ]:



