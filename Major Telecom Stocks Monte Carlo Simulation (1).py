#!/usr/bin/env python
# coding: utf-8

# In[1]:


"Rishabh Shah"
"Monte Carlo Simulation for Telecom Stocks: Verizon, T-Mobile, Sprint, Vodafone, and AT&T"
import pandas as pd   
import numpy as np
import math
from math import *
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Use class monte carlo to initiate the simualtion
class monte_carlo:
    def __init__(self, start, end):
        self.start = start
        self.end = end
#Getting the stock data from Yahoo for dates  we want 
stocks = ["VZ", "T", "TMUS", "S", "VOD"]
data = pd.DataFrame()
for x in stocks:                         
    data[x] = wb.DataReader(x,data_source='yahoo', start='2013-01-01', end='2018-01-01')['Close']
log_returns = np.log(1 + data.pct_change())
log_returns.tail()


# In[3]:


# Take the mean and and variance of the log returns of each stock 
# It will be used in calculating drift 

vz_mean=log_returns['VZ'].mean()
vz_var=log_returns['VZ'].var()
att_mean=log_returns['T'].mean()
att_var=log_returns['T'].var()
tmobile_mean=log_returns['TMUS'].mean()
tmobile_var=log_returns['TMUS'].var()
sprint_mean=log_returns['S'].mean()
sprint_var=log_returns['S'].var()
vod_mean=log_returns['VOD'].mean()
vod_var=log_returns['VOD'].var()



# In[4]:


# Subtract the mean from 0.5 times the variane to calculate the drift of each stock 
vz_drift = vz_mean - (0.5 * vz_var)
att_drift = att_mean - (0.5 * att_var)
tmobile_drift = tmobile_mean - (0.5 * tmobile_var)
sprint_drift = sprint_mean - (0.5 * sprint_var)
vod_drift = vod_mean - (0.5 * vod_var)


# In[5]:


# Calculate the log standard returns of each stock 
vz_std=log_returns['VZ'].std()
att_std=log_returns['T'].std()
tmobile_std=log_returns['TMUS'].std()
sprint_std=log_returns['S'].std()
vod_std=log_returns['VOD'].std()


# In[6]:


# Made numpy arrays for drift and standard deviation. 
#will need them later for daily returns randomization
np.array(vz_drift)


# In[7]:


np.array(vz_std)


# In[8]:


norm.ppf(0.95)


# In[9]:


# Make 10,2 random array
# will be used in calculation for daily returns 
x = np.random.rand(10, 2)
x


# In[10]:


norm.ppf(x)


# In[11]:


Z = norm.ppf(np.random.rand(10,2))
Z


# In[12]:


t_intervals = 1000
iterations = 10
vz_returns = np.exp(np.array(vz_drift) + np.array(vz_std) * norm.ppf(np.random.rand(t_intervals, iterations)))
vz_returns


# In[13]:


S0 = data['VZ'].iloc[-1]
vz_price_list = np.zeros_like(vz_returns)
vz_price_list
vz_price_list[0]


# In[14]:


vz_price_list[0] = S0
for t in range(1, t_intervals):
    vz_price_list[t] = vz_price_list[t - 1] * vz_returns[t]
vz_price_list


# In[15]:


plt.figure(figsize=(10,6))
plt.plot(vz_price_list);
plt.title("Verizon Forecast Price")


# In[16]:


t_intervals = 1000
iterations = 10
att_returns = np.exp(np.array(att_drift) + np.array(att_std) * norm.ppf(np.random.rand(t_intervals, iterations)))
att_returns


# In[17]:


S0 = data['T'].iloc[-1]
att_price_list = np.zeros_like(att_returns)
att_price_list
att_price_list[0]


# In[18]:


att_price_list[0] = S0
for t in range(1, t_intervals):
    att_price_list[t] = att_price_list[t - 1] * att_returns[t]
att_price_list


# In[19]:


plt.figure(figsize=(10,6))
plt.plot(att_price_list);
plt.title("AT&T Forecast Price")


# In[20]:


t_intervals = 1000
iterations = 10
tmobile_returns = np.exp(np.array(tmobile_drift) + np.array(tmobile_std) * norm.ppf(np.random.rand(t_intervals, iterations)))
tmobile_returns


# In[21]:


S0 = data['TMUS'].iloc[-1]
tmobile_price_list = np.zeros_like(tmobile_returns)
tmobile_price_list
tmobile_price_list[0]


# In[22]:


tmobile_price_list[0] = S0
for t in range(1, t_intervals):
    tmobile_price_list[t] = tmobile_price_list[t - 1] * tmobile_returns[t]
tmobile_price_list


# In[23]:


plt.figure(figsize=(10,6))
plt.plot(tmobile_price_list);
plt.title("T-Mobile US Forecast Price")


# In[24]:


t_intervals = 1000
iterations = 10
sprint_returns = np.exp(np.array(sprint_drift) + np.array(sprint_std) * norm.ppf(np.random.rand(t_intervals, iterations)))
sprint_returns


# In[25]:


S0 = data['S'].iloc[-1]
sprint_price_list = np.zeros_like(sprint_returns)
sprint_price_list
sprint_price_list[0]


# In[26]:


sprint_price_list[0] = S0
for t in range(1, t_intervals):
    sprint_price_list[t] = sprint_price_list[t - 1] * sprint_returns[t]
plt.figure(figsize=(10,6))
plt.plot(sprint_price_list);
plt.title("Sprint Forecast Price")


# In[27]:


t_intervals = 1000
iterations = 10
vod_returns = np.exp(np.array(vod_drift) + np.array(vod_std) * norm.ppf(np.random.rand(t_intervals, iterations)))
vod_returns


# In[28]:


S0 = data['VOD'].iloc[-1]
vod_price_list = np.zeros_like(vod_returns)
vod_price_list
vod_price_list[0]


# In[29]:


vod_price_list[0] = S0
for t in range(1, t_intervals):
    vod_price_list[t] = vod_price_list[t - 1] * vod_returns[t]
plt.figure(figsize=(10,6))
plt.plot(vod_price_list);
plt.title("VOD Forecast Price")


# In[ ]:




