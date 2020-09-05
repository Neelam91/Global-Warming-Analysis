#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:26:53 2020

@author: neelamswami
"""

# global sea level rise
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_excel('/Users/neelamswami/Personal Project/global_sea-level.xls', sheet_name = 'sea_level')

# creat two sample population by spliting the data

df_1 = df[90:121]

df_2 = df[121:134]

#summary statistics of both sample population
df_1_sa = df_1['Adjusted Sea Level (inches)'].mean()
# mean = 5.953
df_1_var = df_1['Adjusted Sea Level (inches)'].var()
#variance = 0.414
df_2_sa = df_2['Adjusted Sea Level (inches)'].mean()
#mean = 7.37
df_2_var = df_2['Adjusted Sea Level (inches)'].var()
#variance = 3.76

# t-critical value and df
df_1_size = df_1['Adjusted Sea Level (inches)'].size
df_2_size = df_2['Adjusted Sea Level (inches)'].size

degreeF = (df_1_size -1) + (df_2_size -1 )-2
#degreeF = 43

#t-critical value
from scipy.stats import t
# define probability
p = 0.95
# retrieve value <= probability
t_critical_value = t.ppf(p, degreeF)
print(t_critical_value)
#t-critical value = 1.681


#import t-stat module
import scipy.stats as stats
t_stat, p_val = stats.ttest_ind(df_1['Adjusted Sea Level (inches)'], df_2['Adjusted Sea Level (inches)'], equal_var=False)

#t-statistics = -2.839 and p-value = 0.011
#plot a time series graph of rate of Arctic ice melting

x1 = np.array(df_1.Year)
x2 = np.array(df_2.Year)

y1 = np.array(df_1['Adjusted Sea Level (inches)'])
y2 = np.array(df_2['Adjusted Sea Level (inches)'])

plt.plot(x1, y1, label = 'first plot')
plt.plot(x2,y2, label = 'second plot')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Average sea level rise in inches')
plt.savefig('/Users/neelamswami/Personal Project/sea_level_rise.png')