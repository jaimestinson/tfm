# MLP bueno
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

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

# plt.plot(np.array(df['Day sin'])[:25])
# plt.plot(np.array(df['Day cos'])[:25])
# plt.xlabel('Time [h]')
# plt.title('Time of day signal')

# Training, Validation and Test Split
n = len(df)

# Training set
X_train = data_x[0:int(n*0.7)]
y_train = data_y[0:int(n*0.7)]

# Validation set
X_val = data_x[int(n*0.7):int(n*0.85)]
y_val = data_y[int(n*0.7):int(n*0.85)]

# Test set
X_test = data_x[int(n*0.85):]
y_test = data_y[int(n*0.85):]

# DataFrame conversion
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_val = pd.DataFrame(X_val)
y_val = pd.DataFrame(y_val)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)






#############################################
# Regressive predictors

# For non vectorized models

#################### TRAINING
# Demand

d = []
e = []
d_t1 = []
e_t1 = []
d_t2 = []
e_t2 = []
d_t3 = []
e_t3 = []
d_t24 = []
e_t24 = []
d_t168 = []
e_t168 = []

train_hours = int(0.7*len(df))+2
validation_hours = int(0.15*len(df))
test_hours = int(0.15*len(df))

train_hours+validation_hours+test_hours == len(df)

len(df)
for i in range(167, train_hours):
    
    d.append(df.demanda_p48[i])
    e.append(df.eolica_p48[i])
    
    d_t1.append(df.demanda_p48[i-1])
    e_t1.append(df.eolica_p48[i-1])
    
    d_t2.append(df.demanda_p48[i-2])
    e_t2.append(df.eolica_p48[i-2])
    
    d_t3.append(df.demanda_p48[i-3])
    e_t3.append(df.eolica_p48[i-3])    
    
    d_t24.append(df.demanda_p48[i-24])
    e_t24.append(df.eolica_p48[i-24])  
    
    d_t168.append(df.demanda_p48[i-168])
    e_t168.append(df.eolica_p48[i-168])
    
    
d = np.array(d)
e = np.array(e)
d_t1 = np.array(d_t1)
e_t1 = np.array(e_t1)
d_t2 = np.array(d_t2)
e_t2 = np.array(e_t2)
d_t3 = np.array(d_t3)
e_t3 = np.array(e_t3)
d_t24 = np.array(d_t24)
e_t24 = np.array(e_t24)
d_t168 = np.array(d_t168)
e_t168 = np.array(e_t168)


########### VALIDATION

d_val = []
e_val = []
d_t1_val = []
e_t1_val = []
d_t2_val = []
e_t2_val = []
d_t3_val = []
e_t3_val = []
d_t24_val = []
e_t24_val = []
d_t168_val = []
e_t168_val = []

train_hours = int(0.7*len(df))+2
validation_hours = int(0.15*len(df))
test_hours = int(0.15*len(df))

train_hours+validation_hours+test_hours == len(df)

len(df)
for i in range(train_hours, -validation_hours):
    
    d_val.append(df.demanda_p48[i])
    e_val.append(df.eolica_p48[i])
    
    d_t1_val.append(df.demanda_p48[i-1])
    e_t1_val.append(df.eolica_p48[i-1])
    
    d_t2_val.append(df.demanda_p48[i-2])
    e_t2_val.append(df.eolica_p48[i-2])
    
    d_t3_val.append(df.demanda_p48[i-3])
    e_t3_val.append(df.eolica_p48[i-3])    
    
    d_t24_val.append(df.demanda_p48[i-24])
    e_t24_val.append(df.eolica_p48[i-24])  
    
    d_t168_val.append(df.demanda_p48[i-168])
    e_t168_val.append(df.eolica_p48[i-168])
    
    
d_val = np.array(d_val)
e_val = np.array(e_val)
d_t1_val = np.array(d_t1_val)
e_t1_val = np.array(e_t1_val)
d_t2_val = np.array(d_t2_val)
e_t2_val = np.array(e_t2_val)
d_t3_val = np.array(d_t3_val)
e_t3_val = np.array(e_t3_val)
d_t24_val = np.array(d_t24_val)
e_t24_val = np.array(e_t24_val)
d_t168_val = np.array(d_t168_val)
e_t168_val = np.array(e_t168_val)





###########################################################
# Vectorized models
train_days = int(len(df)*0.7) // 24 + int(len(df)*0.7) % 24
train_days

d_v = []
e_v = []
d_v_t1 = []
e_v_t1 = []
d_v_t2 = []
e_v_t2 = []
d_v_t7 = []
e_v_t7 = []

for i in range(7, train_days): #day i
    
    d_v.insert(i, df.demanda_p48[i*24: (i+1)*24])
    e_v.insert(i, df.eolica_p48[i*24: (i+1)*24])
               
    d_v_t1.insert(i, df.demanda_p48[(i-1)*24: i*24])
    e_v_t1.insert(i, df.eolica_p48[(i-1)*24: i*24])

    d_v_t2.insert(i, df.demanda_p48[(i-2)*24: (i-1)*24])
    e_v_t2.insert(i, df.eolica_p48[(i-2)*24: (i-1)*24])
    
    d_v_t7.insert(i, df.demanda_p48[(i-7)*24: (i-6)*24])
    e_v_t7.insert(i, df.eolica_p48[(i-7)*24: (i-6)*24])
    
    
d_v = np.array(d_v)
e_v = np.array(e_v)
d_v_t1 = np.array(d_v_t1)
e_v_t1 = np.array(e_v_t1)
d_v_t2 = np.array(d_v_t2)
e_v_t2 = np.array(e_v_t2)
d_v_t7 = np.array(d_v_t7)
e_v_t7 = np.array(e_v_t7)


np.info(d_v)

e_v[0][3] # dia 8 ene a las 3h
e_v[10][5] # dia 18 ene a las 5h

d_v_t7[5][23] # dia a las 23h




df.index
df.index.freq = 'H'

# Evolution of the features over time:
plot_cols = ['demanda_p48', 'eolica_p48', 'precio_spot']
plot_features = df[plot_cols]
_ = plot_features.plot(subplots=True)

# Correlation between the exogenous variables
sns.heatmap(df.corr())

# ACF and PACF plots
plot_acf(df.precio_spot, lags = 50)
plot_pacf(df.precio_spot, lags = 50)

# Defining the exogenous variables and the target 
# Non vectorized models
# We take demand and eolic from that hour, the previous 3 hours, one day before and one week before
X_train = [d, e, d_t1, d_t2, e_t2, d_t3, e_t3, d_t24, e_t24, d_t168, e_t168]

# Vectorized models
# We take demand and eolic from that day, the previous two days and one week before
predictors_v = [d_v, e_v, d_v_t1, e_v_t1, d_v_t2, e_v_t2, d_v_t7, e_v_t7]

target_train = df.precio_spot[167:train_hours]
target_val = df.precio_spot[train_hours:-validation_hours]
target_test = df.precio_spot[-validation_hours:]


# Training set
X_train.info()
y_train = pd.DataFrame(target_train)

# # Validation set
# X_val = data_x[int(n*0.7):int(n*0.85)]
# y_val = data_y[int(n*0.7):int(n*0.85)]

# Test set
# X_test = data_x[int(n*0.85):]
# y_test = data_y[int(n*0.85):]

# DataFrame conversion
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_val = pd.DataFrame(X_val)
y_val = pd.DataFrame(y_val)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

# Training set
print('X_train Shape: ', X_train.shape)
print('y_train Shape: ', target_train.shape)

# Validation set
print('X_val Shape: ', X_val.shape)
print('y_val Shape: ', y_val.shape)

# Test set
print('x_test Shape: ', X_test.shape)
print('y_test Shape: ', y_test.shape)

df_mlp = MLPRegressor(hidden_layer_sizes=(7), max_iter=2000,
                               activation='relu',solver='adam', learning_rate='adaptive',
                               learning_rate_init= 0.0001, shuffle = False, alpha = 0.0005, random_state=1)

df_mlp.fit(X_train.transpose(), target_train)

############ VALIDATION



for i in range(train_hours, -validation_hours):
    df_mlp()

predictions_val = df_mlp.predict(X_val)
predictions_val = pd.DataFrame(predictions_val)









########### TEST 

d_test = []
e_test = []
d_t1_test = []
e_t1_test = []
d_t2_test = []
e_t2_test = []
d_t3_test = []
e_t3_test = []
d_t24_test = []
e_t24_test = []
d_t168_test = []
e_t168_test = []

train_hours = int(0.7*len(df))+2
validation_hours = int(0.15*len(df))
test_hours = int(0.15*len(df))

train_hours+validation_hours+test_hours == len(df)

len(df)
for i in range(train_hours, -validation_hours):
    
    d_val.append(df.demanda_p48[i])
    e_val.append(df.eolica_p48[i])
    
    d_t1_val.append(df.demanda_p48[i-1])
    e_t1_val.append(df.eolica_p48[i-1])
    
    d_t2_val.append(df.demanda_p48[i-2])
    e_t2_val.append(df.eolica_p48[i-2])
    
    d_t3_val.append(df.demanda_p48[i-3])
    e_t3_val.append(df.eolica_p48[i-3])    
    
    d_t24_val.append(df.demanda_p48[i-24])
    e_t24_val.append(df.eolica_p48[i-24])  
    
    d_t168_val.append(df.demanda_p48[i-168])
    e_t168_val.append(df.eolica_p48[i-168])
    
    
d_val = np.array(d_val)
e_val = np.array(e_val)
d_t1_val = np.array(d_t1_val)
e_t1_val = np.array(e_t1_val)
d_t2_val = np.array(d_t2_val)
e_t2_val = np.array(e_t2_val)
d_t3_val = np.array(d_t3_val)
e_t3_val = np.array(e_t3_val)
d_t24_val = np.array(d_t24_val)
e_t24_val = np.array(e_t24_val)
d_t168_val = np.array(d_t168_val)
e_t168_val = np.array(e_t168_val)









predictions_val = df_mlp.predict(X_val)
predictions_val = pd.DataFrame(predictions_val)
predictions_test = df_mlp.predict(X_test)
predictions_test = pd.DataFrame(predictions_test)


y_val['Prediction'] = predictions_val.values
y_val['Prediction'] = predictions_val.to_numpy()

y_test['Prediction'] = predictions_test.values
y_test['Prediction'] = predictions_test.to_numpy()

from sklearn.metrics import mean_squared_error

RMSE_val = math.sqrt(mean_squared_error(y_val.precio_spot, y_val.Prediction))

RMSE_test = math.sqrt(mean_squared_error(y_test.precio_spot, y_test.Prediction))

RMSE_val
RMSE_test

def mean_absolute_percentage_error (y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))

MAPE_val = mean_absolute_percentage_error(y_val.precio_spot, y_val.Prediction)
MAPE_test = mean_absolute_percentage_error(y_test.precio_spot, y_test.Prediction)
MAPE_val
MAPE_test


y_train.info()
y_test.info()



fig, ax = plt.subplots(figsize=(9, 4))
plt.xlabel('Date')
plt.ylabel('EP')
plt.title('EPF')
y_train.precio_spot.plot(ax=ax,  color = "black", label='Training')
y_val.precio_spot.plot(ax=ax,  color = "blue", label='Validation')
y_val.Prediction.plot(ax=ax,  color = "green", label='Predictions_val')
y_test.precio_spot.plot(ax=ax, color = "red", label='Test')
y_test.Prediction.plot(ax=ax, color='green', label='Predictions_test')
ax.legend()
