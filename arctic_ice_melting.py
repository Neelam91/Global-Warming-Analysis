#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 08:50:16 2020

@author: neelamswami
"""

#Arctic Sea Ice sheet melting rate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read the excel data in python
df = pd.read_excel('/Users/neelamswami/Personal Project/Arctic_ice_melting.xlsx' )

# clean the data and make two sample population for t-test

df_1 = df[:22]
df_1 = df_1.loc[:,['Year','extent (in Million square KM)']]

df_2 = df[22:38]
df_2 = df_2.loc[:,['Year','extent (in Million square KM)']]


#rename column names in order to run summary statistics
df_1.columns = ['Year', 'Extent Melt (in Million square KM)']
df_2.columns = ['Year', 'Extent Melt (in Million square KM)']

#summary stats
df_1_sa = df_1['Extent Melt (in Million square KM)'].mean()

#mean = 6.91
df_1_var = df_1['Extent Melt(in Million square KM)'].var()

df_2_sa = df_2['Extent Melt (in Million square KM)'].mean()
#mean = 5.18
df_2_var = df_2['Extent Melt(in Million square KM)'].var()


#import t-stat module
import scipy.stats as stats
t_stat, p_val = stats.ttest_ind(df_1['Extent Melt (in Million square KM)'], df_2['Extent Melt (in Million square KM)'], equal_var=False)

#t-stat = 7.549 and p-value = 0.0000


# find DF to get t-critical value
df_1_degf = df_1['Extent Melt (in Million square KM)'].size
df_2_degf = df_2['Extent Melt (in Million square KM)'].size

degreeF = (df_1_degf -1) + (df_2_degf-1) -2
#degree of freedom (df) = 34

#t-critical value
from scipy.stats import t
# define probability
p = 0.95
degreeF = 34
# retrieve value <= probability
t_critical_value = t.ppf(p, degreeF)
print(t_critical_value)

#t_critical = 1.690
#plot a time series graph of rate of Arctic ice melting

x1 = np.array(df_1.Year)
x2 = np.array(df_2.Year)

y1 = np.array(df_1['Extent Melt (in Million square KM)'])
y2 = np.array(df_2['Extent Melt (in Million square KM)'])

plt.plot(x1, y1, label = 'first plot')
plt.plot(x2,y2, label = 'second plot')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Average rate of ice melting(in Million Square KM)')

plt.savefig('/Users/neelamswami/Personal Project/Arctic_ice_melting.png')





