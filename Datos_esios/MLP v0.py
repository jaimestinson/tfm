# MLP 26 november

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import datetime
import math

import IPython
import IPython.display

import matplotlib as mpl
import seaborn as sns
import tensorflow as tf

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
 
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor



df = pd.read_csv('dataset.csv',sep = ';', nrows = 17544)

a = [str(date) for date in df['datetime']]
a = [date[:-6] for date in a]
date_time = pd.to_datetime(a)

df.loc[:,['datetime']] = date_time
df = df.set_index('datetime') 

timestamp_s = date_time.map(pd.Timestamp.timestamp)
day = 24*60*60
week = 7*day
year = (365.2425)*day

df['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['week_sin'] = np.sin(timestamp_s * (2 * np.pi / week))
df['week_cos'] = np.cos(timestamp_s * (2 * np.pi / week))
df['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))



# Evolution of the features over time:
plot_cols = ['demanda_p48', 'eolica_p48', 'precio_spot']
plot_features = df[plot_cols]
_ = plot_features.plot(subplots=True)


#############################################
# Regressive predictors

# For non vectorized models

#################### TRAINING
# Demand

d_t1 = []
e_t1 = []
d_t2 = []
e_t2 = []
d_t3 = []
e_t3 = []
d_t4 = []
e_t4 = []
d_t24 = []
e_t24 = []
d_t168 = []
e_t168 = []
day_sin_train = []
day_cos_train = []
week_sin_train = []
week_cos_train = []
year_sin_train = []
year_cos_train = []


train_hours = int(0.7*len(df))+2
validation_hours = int(0.15*len(df))
test_hours = int(0.15*len(df))

train_hours+validation_hours+test_hours == len(df)

len(df)
for i in range(167, train_hours):
    
    d_t1.append(df.demanda_p48[i-1])
    e_t1.append(df.eolica_p48[i-1])
    
    d_t2.append(df.demanda_p48[i-2])
    e_t2.append(df.eolica_p48[i-2])
    
    d_t3.append(df.demanda_p48[i-3])
    e_t3.append(df.eolica_p48[i-3])  
    
    d_t4.append(df.demanda_p48[i-4])
    e_t4.append(df.eolica_p48[i-4]) 
    
    d_t24.append(df.demanda_p48[i-24])
    e_t24.append(df.eolica_p48[i-24])  
    
    d_t168.append(df.demanda_p48[i-168])
    e_t168.append(df.eolica_p48[i-168])
    
    day_sin_train.append(df.day_sin[i])
    day_cos_train.append(df.day_cos[i])
    week_sin_train.append(df.week_sin[i])
    week_cos_train.append(df.week_cos[i])
    year_sin_train.append(df.year_sin[i])
    year_cos_train.append(df.year_cos[i])
    
    
d_t1 = np.array(d_t1)
e_t1 = np.array(e_t1)
d_t2 = np.array(d_t2)
e_t2 = np.array(e_t2)
d_t3 = np.array(d_t3)
e_t3 = np.array(e_t3)
d_t4 = np.array(d_t4)
e_t4 = np.array(e_t4)
d_t24 = np.array(d_t24)
e_t24 = np.array(e_t24)
d_t168 = np.array(d_t168)
e_t168 = np.array(e_t168)
day_sin_train = np.array(day_sin_train)
day_cos_train = np.array(day_cos_train)
week_sin_train = np.array(week_sin_train)
week_cos_train = np.array(week_cos_train)
year_sin_train = np.array(year_sin_train)
year_cos_train = np.array(year_cos_train)



########### VALIDATION

d_t1_val = []
e_t1_val = []
d_t2_val = []
e_t2_val = []
d_t3_val = []
e_t3_val = []
d_t4_val = []
e_t4_val = []
d_t24_val = []
e_t24_val = []
d_t168_val = []
e_t168_val = []
day_sin_val = []
day_cos_val = []
week_sin_val = []
week_cos_val = []
year_sin_val = []
year_cos_val = []


for i in range(train_hours, train_hours + validation_hours):
    
    d_t1_val.append(df.demanda_p48[i-1])
    e_t1_val.append(df.eolica_p48[i-1])
    
    d_t2_val.append(df.demanda_p48[i-2])
    e_t2_val.append(df.eolica_p48[i-2])
    
    d_t3_val.append(df.demanda_p48[i-3])
    e_t3_val.append(df.eolica_p48[i-3])  
    
    d_t4_val.append(df.demanda_p48[i-4])
    e_t4_val.append(df.eolica_p48[i-4])  
    
    d_t24_val.append(df.demanda_p48[i-24])
    e_t24_val.append(df.eolica_p48[i-24])  
    
    d_t168_val.append(df.demanda_p48[i-168])
    e_t168_val.append(df.eolica_p48[i-168])
    
    day_sin_val.append(df.day_sin[i])
    day_cos_val.append(df.day_cos[i])
    week_sin_val.append(df.week_sin[i])
    week_cos_val.append(df.week_cos[i])
    year_sin_val.append(df.year_sin[i])
    year_cos_val.append(df.year_cos[i])
    
    
    
d_t1_val = np.array(d_t1_val)
e_t1_val = np.array(e_t1_val)
d_t2_val = np.array(d_t2_val)
e_t2_val = np.array(e_t2_val)
d_t3_val = np.array(d_t3_val)
e_t3_val = np.array(e_t3_val)
d_t4_val = np.array(d_t4_val)
e_t4_val = np.array(e_t4_val)
d_t24_val = np.array(d_t24_val)
e_t24_val = np.array(e_t24_val)
d_t168_val = np.array(d_t168_val)
e_t168_val = np.array(e_t168_val)
day_sin_val = np.array(day_sin_val)
day_cos_val = np.array(day_cos_val)
week_sin_val = np.array(week_sin_val)
week_cos_val = np.array(week_cos_val)
year_sin_val = np.array(year_sin_val)
year_cos_val = np.array(year_cos_val)

########### TEST


d_t1_test = []
e_t1_test = []
d_t2_test = []
e_t2_test = []
d_t3_test = []
e_t3_test = []
d_t4_test = []
e_t4_test = []
d_t24_test = []
e_t24_test = []
d_t168_test = []
e_t168_test = []
day_sin_test = []
day_cos_test = []
week_sin_test = []
week_cos_test = []
year_sin_test = []
year_cos_test = []


for i in range(train_hours+validation_hours, len(df)):
    
    d_t1_test.append(df.demanda_p48[i-1])
    e_t1_test.append(df.eolica_p48[i-1])
    
    d_t2_test.append(df.demanda_p48[i-2])
    e_t2_test.append(df.eolica_p48[i-2])
    
    d_t3_test.append(df.demanda_p48[i-3])
    e_t3_test.append(df.eolica_p48[i-3])    
    
    d_t4_test.append(df.demanda_p48[i-4])
    e_t4_test.append(df.eolica_p48[i-4])    
    
    d_t24_test.append(df.demanda_p48[i-24])
    e_t24_test.append(df.eolica_p48[i-24])  
    
    d_t168_test.append(df.demanda_p48[i-168])
    e_t168_test.append(df.eolica_p48[i-168])
    
    day_sin_test.append(df.day_sin[i])
    day_cos_test.append(df.day_cos[i])
    week_sin_test.append(df.week_sin[i])
    week_cos_test.append(df.week_cos[i])
    year_sin_test.append(df.year_sin[i])
    year_cos_test.append(df.year_cos[i])
    
    
d_t1_test = np.array(d_t1_test)
e_t1_test = np.array(e_t1_test)
d_t2_test = np.array(d_t2_test)
e_t2_test = np.array(e_t2_test)
d_t3_test = np.array(d_t3_test)
e_t3_test = np.array(e_t3_test)
d_t4_test = np.array(d_t4_test)
e_t4_test = np.array(e_t4_test)
d_t24_test = np.array(d_t24_test)
e_t24_test = np.array(e_t24_test)
d_t168_test = np.array(d_t168_test)
e_t168_test = np.array(e_t168_test)
day_sin_test = np.array(day_sin_test)
day_cos_test = np.array(day_cos_test)
week_sin_test = np.array(week_sin_test)
week_cos_test = np.array(week_cos_test)
year_sin_test = np.array(year_sin_test)
year_cos_test = np.array(year_cos_test)



# Defining the exogenous variables and the target 
# Non vectorized models
# We take demand and eolic from that hour, the previous 3 hours, one day before and one week before
X_train = [d_t1, e_t1, d_t2, e_t2, d_t3, e_t3, d_t4, e_t4, d_t24, e_t24, d_t168, e_t168, day_sin_train, day_cos_train, week_sin_train, week_cos_train, year_sin_train, year_cos_train]
X_train.d_t1

X_train = pd.DataFrame(data={'d_t1':d_t1, 'e_t1':e_t1, 'd_t2':d_t2, 'e_t2':e_t2, 'd_t3':d_t3, 'e_t3':e_t3, 'd_t4':d_t4, 'e_t4':e_t4, 'd_t24':d_t24, 'e_t24':e_t24, 'd_t168':d_t168, 'e_t168':e_t168, 'day_sin':day_sin_train, 'day_cos':day_cos_train, 'week_sin':week_sin_train, 'week_cos':week_cos_train, 'year_sin':year_sin_train, 'year_cos':year_cos_train})
X_test = pd.DataFrame(data={'d_t1':d_t1_test, 'e_t1':e_t1_test, 'd_t2':d_t2_test, 'e_t2':e_t2_test, 'd_t3':d_t3_test, 'e_t3':e_t3_test, 'd_t4':d_t4_test, 'e_t4':e_t4_test, 'd_t24':d_t24_test, 'e_t24':e_t24_test, 'd_t168':d_t168_test, 'e_t168':e_t168_test, 'day_sin':day_sin_test, 'day_cos':day_cos_test, 'week_sin':week_sin_test,'week_cos': week_cos_test, 'year_sin':year_sin_test, 'year_cos':year_cos_test})


X_val = [d_t1_val, e_t1_val, d_t2_val, e_t2_val, d_t3_val, e_t3_val, d_t4_val, e_t4_val, d_t24_val, e_t24_val, d_t168_val, e_t168_val, day_sin_val, day_cos_val, week_sin_val, week_cos_val, year_sin_val, year_cos_val]
X_test = [d_t1_test, e_t1_test, d_t2_test, e_t2_test, d_t3_test, e_t3_test, d_t4_test, e_t4_test, d_t24_test, e_t24_test, d_t168_test, e_t168_test, day_sin_test, day_cos_test, week_sin_test, week_cos_test, year_sin_test, year_cos_test]

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

X_train.shape

X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[0], 1))
X_val = np.reshape(X_val, (X_val.shape[1], X_val.shape[0], 1))
X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[0], 1))


# X_train = pd.DataFrame(X_train)

y_train = np.array(df.precio_spot[167: train_hours])
y_val = np.array(df.precio_spot[train_hours: -test_hours])
y_test = np.array(df.precio_spot[-test_hours:])

len(y_train) == len(X_train)
len(y_val) == len(X_val)
len(y_test) == len(X_test)

# MLP
df_mlp = MLPRegressor(hidden_layer_sizes=(2), max_iter=5000,
                               activation='relu',solver='adam', learning_rate='adaptive',
                               learning_rate_init= 0.0001, shuffle = False, alpha = 0.0005, random_state=1)

df_mlp.fit(X_train, y_train)


# evaluate the model on the second set of data

train_predictions = df_mlp.predict(X_train)

test_predictions = df_mlp.predict(X_test)


plt.plot(y_test, color = 'blue')
plt.plot(test_predictions, color = 'green')


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_predictions))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_predictions))  
print('Root Mean Squared Error:', np.mean(np.sqrt(metrics.mean_squared_error(y_test, test_predictions))))




# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'hidden_layer_sizes': [2],
    'max_iter': [50, 100, 500, 1000, 2000],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
    'learning_rate': ['constant','adaptive'],
}



grid = GridSearchCV(df_mlp, param_grid, n_jobs= -1, cv=5)
grid.fit(X_train, y_train)





































































































