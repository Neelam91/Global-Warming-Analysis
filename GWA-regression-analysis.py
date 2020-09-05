#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:27:23 2020

@author: neelamswami
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import all the Dependent x-variables

df_co2 = pd.read_excel('/Users/neelamswami/Personal Project/rawdata..xls', sheet_name = 'Co2_Emission')

df_Deforestation = pd.read_excel('/Users/neelamswami/Personal Project/rawdata..xls', sheet_name = 'Deforestation_data')

df_Renewable_Energy = pd.read_excel('/Users/neelamswami/Personal Project/rawdata..xls', sheet_name = 'Renewable_energy_share')

df_ghg_emission = pd.read_excel('/Users/neelamswami/Personal Project/rawdata..xls', sheet_name = 'ghg_emissions')

df_fossil_fuel = pd.read_excel('/Users/neelamswami/Personal Project/rawdata..xls', sheet_name = 'fossil_fuel_consumption')

df_Livestock = pd.read_excel('/Users/neelamswami/Personal Project/rawdata..xls', sheet_name = 'Livestock_Production')

df_Agriculture = pd.read_excel('/Users/neelamswami/Personal Project/rawdata..xls', sheet_name = 'Agriculture_data')

#cleanup the data and select data for required year
df_co2_new = df_co2[11:58]

#cleanup Deforestation data
df_Deforestation = df_Deforestation[9:]


#average of country wise data
df_Renewable_Energy = df_Renewable_Energy.groupby("Year")[ 'Year','Renewables_energy_share (%)'].mean()
#remove indexing error
df_Renewable_Energy = df_Renewable_Energy.rename(columns={'Year':'Year_1'})

df_Renewable_Energy = df_Renewable_Energy.reset_index()

df_Renewable_Energy = df_Renewable_Energy.drop('Year_1',1)

#clean uo GHG data
df_ghg_emission = df_ghg_emission.groupby('Year')['Year', 'GHG emissions per capita (tonnes carbon dioxide equivalents (CO‚ÇÇe))'].mean()

#rename Year column to year_1 to avoid error with year in index
df_ghg_emission = df_ghg_emission.rename(columns={'Year':'Year_1'})

df_ghg_emission = df_ghg_emission.reset_index()

df_ghg_emission= df_ghg_emission.drop('Year_1',1)


#clean up fossil fuel data
df_fossil_fuel = df_fossil_fuel[22:]
df_fossil_fuel_new = df_fossil_fuel.loc[:,['Total_fossil_fuel_consumption','Year']]


#clean up livestock data
df_Livestock = df_Livestock.T
df_Livestock_t = df_Livestock.drop(df_Livestock.index[[0, 1, 2,3]])
df_Livestock_t['Average_livestock_production'] = df_Livestock_t.mean(axis = 1)
df_Livestock_t['Year'] = df_Livestock_t.index

#reseting index from year to number
df_Livestock_new = df_Livestock_t.reset_index(inplace = True)

df_Livestock_new = df_Livestock_t[9:]
df_Livestock_new = df_Livestock_new.drop('level_0',1)
df_Livestock_new = df_Livestock_new.drop('index',1)
df_Livestock_new = df_Livestock_new.loc[:,['Average_livestock_production','Year']]


#clean up agriculture data
df_agri_t = df_Agriculture.T

#drop extra rows from dataset
df_agri_t = df_agri_t.drop(df_agri_t.index[[0, 1, 2,3]])

df_agri_withoutmissing = df_agri_t.fillna(df_agri_t.mean())

#add a new column with average value per country per year
df_agri_withoutmissing['Average_agricultural_land'] = df_agri_withoutmissing.mean(axis = 1)

#create a new column with year
df_agri_withoutmissing['Year'] = df_agri_withoutmissing.index

#reseting index from year to number

df_agri_withoutmissing = df_agri_withoutmissing.reset_index(inplace = True)
df_agriculture_new = df_agri_withoutmissing.drop('index',1)

df_agriculture_new = df_agriculture_new[9:56]

df_agriculture_new = df_agriculture_new.loc[:,['Average_agricultural_land','Year']]


#Merge all the independent variables

total_X1 = pd.merge(df_co2_new, df_Deforestation, on=['Year'])
total_X2 = pd.merge(total_X1,df_Renewable_Energy, on = ['Year'])


total_X3 = pd.merge(total_X2, df_fossil_fuel, on = ['Year'])
total_X4 = pd.merge(total_X3, df_ghg_emission, on = ['Year'])
total_X5 = pd.merge(total_X4, df_agriculture_new, on = ['Year'])

total_X6 = pd.merge(total_X5, df_Livestock_new, on = ['Year'])


#Final x-variables
x_variables = total_X6.loc[:,['CO2 Emission average (ppm)', 'Total Wood Production','Renewables_energy_share (%)','Total_fossil_fuel_consumption','GHG emissions per capita (tonnes carbon dioxide equivalents (CO‚ÇÇe))','Average_agricultural_land', 'Average_livestock_production']]

#Import Y-variable
y_variables = pd.read_excel('/Users/neelamswami/Personal Project/global_temp_data.xlsx', sheet_name = 'Global_temp' )

y_variable = y_variables[90:]

y_variable = y_variable.loc[:,['Mean_gtemp']]


#Multivariate Regression process

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x_variables, y_variable, test_size=1/3, random_state=0)


#reset indexes to avoid errors in stasmodel run
x_train = x_train.reset_index()
x_train = x_train.drop('index',1)

y_train = y_train.reset_index()
y_train = y_train.drop('index',1)

#statsmodel import to get summary statistics of training dataset to check the statistics of the model
import statsmodels.api as sm
x_train['intercept'] = 1.0
linear_model = sm.OLS(y_train,x_train)
results = linear_model.fit()
print(results.summary2())


'''
                                       Results: Ordinary least squares
=======================================================================================================================
Model:                                 OLS                               Adj. R-squared:                      0.940    
Dependent Variable:                    Mean_gtemp                        AIC:                                 -72.9278 
Date:                                  2020-08-29 15:15                  BIC:                                 -61.4559 
No. Observations:                      31                                Log-Likelihood:                      44.464   
Df Model:                              7                                 F-statistic:                         68.38    
Df Residuals:                          23                                Prob (F-statistic):                  6.93e-14 
R-squared:                             0.954                             Scale:                               0.0044805
-----------------------------------------------------------------------------------------------------------------------
                                                                       Coef.   Std.Err.    t    P>|t|   [0.025   0.975]
-----------------------------------------------------------------------------------------------------------------------
CO2 Emission average (ppm)                                              0.0364   0.0083  4.3585 0.0002   0.0191  0.0536
Total Wood Production                                                   0.0003   0.0002  1.6670 0.1091  -0.0001  0.0007
Renewables_energy_share (%)                                             0.0703   0.0417  1.6838 0.1057  -0.0161  0.1566
Total_fossil_fuel_consumption                                          -0.0000   0.0000 -0.1576 0.8762  -0.0000  0.0000
GHG emissions per capita (tonnes carbon dioxide equivalents (CO‚ÇÇe))   0.0994   0.0429  2.3155 0.0299   0.0106  0.1882
Average_agricultural_land                                              -0.0425   0.1631 -0.2604 0.7969  -0.3800  0.2950
Average_livestock_production                                           -0.0291   0.0107 -2.7211 0.0122  -0.0513 -0.0070
intercept                                                             -11.0640   5.3474 -2.0691 0.0500 -22.1259 -0.0022
-----------------------------------------------------------------------------------------------------------------------
Omnibus:                               2.205                         Durbin-Watson:                            2.305   
Prob(Omnibus):                         0.332                         Jarque-Bera (JB):                         1.704   
Skew:                                  0.405                         Prob(JB):                                 0.426   
Kurtosis:                              2.186                         Condition No.:                            40513642
=======================================================================================================================



'''

#based on the above results p-value is not significant, so Iam dropping Co2 emission data as it is theoritically correalted with GHG emission data
#correlation checking
#from statsmodels.stats.outliers_influence import variance_inflation_factor
#vif = pd.DataFrame()
#vif['VIF factor'] = [variance_inflation_factor(x_variables.values,i) for i in range(x_variables.shape[1])]
#vif['features'] = x_variables.columns
#vif.round(1)


#recurssive feature elimination method to choose important features only

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

from sklearn.feature_selection import RFE

# Create linear regression object
regressor = LinearRegression()
rfe = RFE(regressor)
rfe = rfe.fit(x_train,y_train)

#Variables checking for feature elimination by checking importance of each feature
dset = pd.DataFrame()
dset['attr'] = x_train.columns
dset['importance'] = rfe.ranking_
dset = dset.sort_values(by = 'importance',ascending = False)
print(dset)


'''
       attr  importance
7                                          intercept           5
3                      Total_fossil_fuel_consumption           4
1                              Total Wood Production           3
6                       Average_livestock_production           2
0                         CO2 Emission average (ppm)           1
2                        Renewables_energy_share (%)           1
4  GHG emissions per capita (tonnes carbon dioxid...           1
5                          Average_agricultural_land           1
/Users/neelamswami/opt/anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py:724: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
  '''

#selecting x_variables with importance 1 after checking feature importance 

x_variables = total_X6.loc[:,['CO2 Emission average (ppm)', 'Renewables_energy_share (%)','GHG emissions per capita (tonnes carbon dioxide equivalents (CO‚ÇÇe))','Average_agricultural_land']]


from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x_variables, y_variable, test_size=1/3, random_state=0)


x_train = x_train.reset_index()
x_train = x_train.drop('index',1)

y_train = y_train.reset_index()
y_train = y_train.drop('index',1)

#statsmodel import to get summary statistics of training dataset for regression model
import statsmodels.api as sm
x_train['intercept'] = 1.0
linear_model = sm.OLS(y_train,x_train)
results = linear_model.fit()
print(results.summary2())




'''   Results: Ordinary least squares
=======================================================================================================================
Model:                                 OLS                               Adj. R-squared:                      0.917    
Dependent Variable:                    Mean_gtemp                        AIC:                                 -64.8389 
Date:                                  2020-08-29 15:28                  BIC:                                 -57.6690 
No. Observations:                      31                                Log-Likelihood:                      37.419   
Df Model:                              4                                 F-statistic:                         83.50    
Df Residuals:                          26                                Prob (F-statistic):                  1.90e-14 
R-squared:                             0.928                             Scale:                               0.0062439
-----------------------------------------------------------------------------------------------------------------------
                                                                       Coef.   Std.Err.    t    P>|t|   [0.025   0.975]
-----------------------------------------------------------------------------------------------------------------------
CO2 Emission average (ppm)                                              0.0102   0.0014  7.2490 0.0000   0.0073  0.0131
Renewables_energy_share (%)                                             0.0214   0.0320  0.6697 0.5089  -0.0444  0.0873
GHG emissions per capita (tonnes carbon dioxide equivalents (CO‚ÇÇe))   0.1179   0.0416  2.8375 0.0087   0.0325  0.2034
Average_agricultural_land                                               0.2373   0.0971  2.4430 0.0217   0.0376  0.4370
intercept                                                             -13.4892   3.8291 -3.5228 0.0016 -21.3601 -5.6184
-----------------------------------------------------------------------------------------------------------------------
Omnibus:                                 0.655                          Durbin-Watson:                            2.178
Prob(Omnibus):                           0.721                          Jarque-Bera (JB):                         0.690
Skew:                                    -0.101                         Prob(JB):                                 0.708
Kurtosis:                                2.297                          Condition No.:                            97487
=======================================================================================================================
* The condition number is large (1e+05). This might indicate             strong multicollinearity or other numerical
problems.
'''


#removing co2 emission dataset to avoid correlation and re-run feature importance

x_variables = total_X6.loc[:,['Total Wood Production','Renewables_energy_share (%)','Total_fossil_fuel_consumption','GHG emissions per capita (tonnes carbon dioxide equivalents (CO‚ÇÇe))','Average_agricultural_land', 'Average_livestock_production']]


'''
 Results: Ordinary least squares
======================================================================================================================
Model:                                OLS                               Adj. R-squared:                      0.895    
Dependent Variable:                   Mean_gtemp                        AIC:                                 -56.2630 
Date:                                 2020-08-29 15:36                  BIC:                                 -46.2251 
No. Observations:                     31                                Log-Likelihood:                      35.131   
Df Model:                             6                                 F-statistic:                         43.78    
Df Residuals:                         24                                Prob (F-statistic):                  9.17e-12 
R-squared:                            0.916                             Scale:                               0.0078402
----------------------------------------------------------------------------------------------------------------------
                                                                       Coef.   Std.Err.    t    P>|t|   [0.025  0.975]
----------------------------------------------------------------------------------------------------------------------
Total Wood Production                                                   0.0000   0.0002  0.2064 0.8383  -0.0004 0.0005
Renewables_energy_share (%)                                             0.0905   0.0549  1.6496 0.1121  -0.0227 0.2037
Total_fossil_fuel_consumption                                           0.0000   0.0000  1.7691 0.0896  -0.0000 0.0000
GHG emissions per capita (tonnes carbon dioxide equivalents (CO‚ÇÇe))   0.0821   0.0565  1.4527 0.1593  -0.0346 0.1988
Average_agricultural_land                                               0.2582   0.1956  1.3204 0.1991  -0.1454 0.6619
Average_livestock_production                                           -0.0131   0.0133 -0.9867 0.3336  -0.0406 0.0143
intercept                                                             -11.6704   7.0712 -1.6504 0.1119 -26.2646 2.9239
----------------------------------------------------------------------------------------------------------------------
Omnibus:                               2.243                         Durbin-Watson:                           2.015   
Prob(Omnibus):                         0.326                         Jarque-Bera (JB):                        1.232   
Skew:                                  -0.086                        Prob(JB):                                0.540   
Kurtosis:                              2.039                         Condition No.:                           40499277
======================================================================================================================
* The condition number is large (4e+07). This might indicate             strong multicollinearity or other numerical
'''
#check importance and summary statistics with important variables
x_variables = total_X6.loc[:,['Renewables_energy_share (%)','GHG emissions per capita (tonnes carbon dioxide equivalents (CO‚ÇÇe))','Average_agricultural_land']]

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x_variables, y_variable, test_size=1/3, random_state=0)


#reset indexes to avoid errors in stasmodel run
x_train = x_train.reset_index()
x_train = x_train.drop('index',1)

y_train = y_train.reset_index()
y_train = y_train.drop('index',1)

#statsmodel import to get summary statistics of training dataset for regression model
import statsmodels.api as sm
x_train['intercept'] = 1.0
linear_model = sm.OLS(y_train,x_train)
results = linear_model.fit()
print(results.summary2())



'''
 Results: Ordinary least squares
========================================================================================================================
Model:                                 OLS                                Adj. R-squared:                       0.758   
Dependent Variable:                    Mean_gtemp                         AIC:                                  -32.5650
Date:                                  2020-08-29 15:40                   BIC:                                  -26.8291
No. Observations:                      31                                 Log-Likelihood:                       20.283  
Df Model:                              3                                  F-statistic:                          32.25   
Df Residuals:                          27                                 Prob (F-statistic):                   4.51e-09
R-squared:                             0.782                              Scale:                                0.018165
------------------------------------------------------------------------------------------------------------------------
                                                                       Coef.   Std.Err.    t    P>|t|   [0.025   0.975] 
------------------------------------------------------------------------------------------------------------------------
Renewables_energy_share (%)                                             0.2128   0.0309  6.8804 0.0000   0.1493   0.2763
GHG emissions per capita (tonnes carbon dioxide equivalents (CO‚ÇÇe))   0.1489   0.0705  2.1118 0.0441   0.0042   0.2936
Average_agricultural_land                                               0.5495   0.1485  3.7004 0.0010   0.2448   0.8543
intercept                                                             -23.7143   6.0718 -3.9057 0.0006 -36.1725 -11.2560
------------------------------------------------------------------------------------------------------------------------
Omnibus:                                 0.297                          Durbin-Watson:                             2.599
Prob(Omnibus):                           0.862                          Jarque-Bera (JB):                          0.251
Skew:                                    0.193                          Prob(JB):                                  0.882
Kurtosis:                                2.787                          Condition No.:                             10009
========================================================================================================================
* The condition number is large (1e+04). This might indicate             strong multicollinearity or other numerical
problems.
'''

#based on the above three x-variables, global temperature is rising as p-value for x variables are statistically significant

#run regression for training data set now that p-value is significant

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


#Now my model is trained with training data set.
#lets test the trained model with test dataset
# Predicting the test set results
y_pred = regressor.predict(x_test)

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))
print('Root_Mean_squared_error is:', rmse)

#http://ijdddonline.com/issues/511_full.pdf
#based on my results of RMSE (0.13) and Adjusted R square (0.76) which is satisfactory not overfitting or underfitting.

#Plot regression outputs with test data
x1 = np.array(y_test.Mean_gtemp)
y1= np.array(y_pred)

plt.plot(x1, y1)



#sea_level_data

# now I would like to analyse new y-variable with the 7 x-variables to check statistical significance.

#import sea level rise data set as y-variable

y_sea = pd.read_excel('/Users/neelamswami/Personal Project/global_sea-level.xls', sheet_name = 'sea_level' )

y_sea = y_sea[90:]

y_sea = y_sea.loc[:,['Adjusted Sea Level (inches)']]


x_variables = total_X6.loc[:,['Total Wood Production','Renewables_energy_share (%)','Total_fossil_fuel_consumption','GHG emissions per capita (tonnes carbon dioxide equivalents (CO‚ÇÇe))','Average_agricultural_land', 'Average_livestock_production']]

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x_variables, y_variable, test_size=1/3, random_state=0)


#reset indexes to avoid errors in stasmodel run
x_train = x_train.reset_index()
x_train = x_train.drop('index',1)

y_train = y_train.reset_index()
y_train = y_train.drop('index',1)

#statsmodel import to get summary statistics of training dataset for regression model
import statsmodels.api as sm
x_train['intercept'] = 1.0
linear_model = sm.OLS(y_train,x_train)
results = linear_model.fit()
print(results.summary2())
'''
Results: Ordinary least squares
======================================================================================================================
Model:                                OLS                               Adj. R-squared:                      0.895    
Dependent Variable:                   Mean_gtemp                        AIC:                                 -56.2630 
Date:                                 2020-08-29 15:49                  BIC:                                 -46.2251 
No. Observations:                     31                                Log-Likelihood:                      35.131   
Df Model:                             6                                 F-statistic:                         43.78    
Df Residuals:                         24                                Prob (F-statistic):                  9.17e-12 
R-squared:                            0.916                             Scale:                               0.0078402
----------------------------------------------------------------------------------------------------------------------
                                                                       Coef.   Std.Err.    t    P>|t|   [0.025  0.975]
----------------------------------------------------------------------------------------------------------------------
Total Wood Production                                                   0.0000   0.0002  0.2064 0.8383  -0.0004 0.0005
Renewables_energy_share (%)                                             0.0905   0.0549  1.6496 0.1121  -0.0227 0.2037
Total_fossil_fuel_consumption                                           0.0000   0.0000  1.7691 0.0896  -0.0000 0.0000
GHG emissions per capita (tonnes carbon dioxide equivalents (CO‚ÇÇe))   0.0821   0.0565  1.4527 0.1593  -0.0346 0.1988
Average_agricultural_land                                               0.2582   0.1956  1.3204 0.1991  -0.1454 0.6619
Average_livestock_production                                           -0.0131   0.0133 -0.9867 0.3336  -0.0406 0.0143
intercept                                                             -11.6704   7.0712 -1.6504 0.1119 -26.2646 2.9239
----------------------------------------------------------------------------------------------------------------------
Omnibus:                               2.243                         Durbin-Watson:                           2.015   
Prob(Omnibus):                         0.326                         Jarque-Bera (JB):                        1.232   
Skew:                                  -0.086                        Prob(JB):                                0.540   
Kurtosis:                              2.039                         Condition No.:                           40499277
======================================================================================================================
* The condition number is large (4e+07). This might indicate             strong multicollinearity or other numerical
problems.
'''

regressor = LinearRegression()
rfe = RFE(regressor)
rfe = rfe.fit(x_train,y_train)

dset = pd.DataFrame()
dset['attr'] = x_train.columns
dset['importance'] = rfe.ranking_
dset = dset.sort_values(by = 'importance',ascending = False)
print(dset)
'''
                attr  importance
6                                          intercept           5
2                      Total_fossil_fuel_consumption           4
0                              Total Wood Production           3
5                       Average_livestock_production           2
1                        Renewables_energy_share (%)           1
3  GHG emissions per capita (tonnes carbon dioxid...           1
4                          Average_agricultural_land           1
'''

#lets check the p-value or weights by using important x-variables for sea level y-variables

x_variables = total_X6.loc[:,['Renewables_energy_share (%)','GHG emissions per capita (tonnes carbon dioxide equivalents (CO‚ÇÇe))','Average_agricultural_land']]
'''
Results: Ordinary least squares
========================================================================================================================
Model:                                 OLS                                Adj. R-squared:                       0.758   
Dependent Variable:                    Mean_gtemp                         AIC:                                  -32.5650
Date:                                  2020-08-29 15:54                   BIC:                                  -26.8291
No. Observations:                      31                                 Log-Likelihood:                       20.283  
Df Model:                              3                                  F-statistic:                          32.25   
Df Residuals:                          27                                 Prob (F-statistic):                   4.51e-09
R-squared:                             0.782                              Scale:                                0.018165
------------------------------------------------------------------------------------------------------------------------
                                                                       Coef.   Std.Err.    t    P>|t|   [0.025   0.975] 
------------------------------------------------------------------------------------------------------------------------
Renewables_energy_share (%)                                             0.2128   0.0309  6.8804 0.0000   0.1493   0.2763
GHG emissions per capita (tonnes carbon dioxide equivalents (CO‚ÇÇe))   0.1489   0.0705  2.1118 0.0441   0.0042   0.2936
Average_agricultural_land                                               0.5495   0.1485  3.7004 0.0010   0.2448   0.8543
intercept                                                             -23.7143   6.0718 -3.9057 0.0006 -36.1725 -11.2560
------------------------------------------------------------------------------------------------------------------------
Omnibus:                                 0.297                          Durbin-Watson:                             2.599
Prob(Omnibus):                           0.862                          Jarque-Bera (JB):                          0.251
Skew:                                    0.193                          Prob(JB):                                  0.882
Kurtosis:                                2.787                          Condition No.:                             10009
========================================================================================================================
* The condition number is large (1e+04). This might indicate             strong multicollinearity or other numerical
problems.
'''

#TIME Searies plot between average sea-level data and renewable energy share as it is showing a positive trend 

y_variable = y_variables[90:134]
Renewable_energy_data = df_Renewable_Energy[5:52]

#
x1 = np.array(Renewable_energy_data.Year)
x2 = np.array(y_variable.Year)

y1 = np.array(Renewable_energy_data['Renewables_energy_share (%)'])
y2 = np.array(y_variable['Adjusted Sea Level (inches)'])

plt.plot(x1, y1, label = 'Renewable_energy')
plt.plot(x2,y2, label = 'Sea level')
plt.legend()
plt.xlabel('Year')
plt.ylabel('renewable energy vs sea level rise')

plt.savefig('/Users/neelamswami/Personal Project/Arctic_ice_melting.png')