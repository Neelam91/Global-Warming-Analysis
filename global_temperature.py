#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 09:35:53 2020

@author: neelamswami
"""

#global Temperature Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# lets read the Global Temperature excel data in python

df = pd.read_excel('/Users/neelamswami/Personal Project/global_temp_data.xlsx', sheet_name = 'Global_temp')

df_1 = df[90:121]

df_2 = df[121:]


#summary stats
df_1_sa = df_1['Mean_gtemp'].mean()

#mean = 0.24
df_1_var = df_1['Mean_gtemp'].var()
#df_1 varince = 0.036
df_2_sa = df_2['Mean_gtemp'].mean()
#mean = 0.67
df_2_var = df_2['Mean_gtemp'].var()
#dr_2 variance = 0.014


#import t-stat module
import scipy.stats as stats
t_stat, p_val = stats.ttest_ind(df_1['Mean_gtemp'], df_2['Mean_gtemp'], equal_var=False)

# t-statistics = -9.707 and p-value = 0.000002, degree of freedom = 43


#t-critical value
from scipy.stats import t
# define probability
p = 0.95
degreeF = 43
# retrieve value <= probability
t_critical_value = t.ppf(p, degreeF)
print(t_critical_value)
#t-critical = 1.681

# Now, lets plot a time series graph

df_graph = pd.read_excel('/Users/neelamswami/Personal Project/global_temp_data.xlsx', sheet_name = 'Global_temp')
x1 = np.array(df_1.Year)
df_graph = df_graph[99:]

x= np.array(df_graph.Year)

y = np.array(df_graph['Mean_gtemp'])


plt.plot(x, y)
plt.title ('Average Global Temperature in Degree Celsius')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Average Temp. anomaly in degree celsius')

plt.savefig('/Users/neelamswami/Personal Project/global_temp.png')


