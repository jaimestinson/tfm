# MLP hora a hora (non vectorized)

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

df = pd.read_csv('dataset.csv',sep = ';', nrows = 17545)

a = [str(date) for date in df['datetime']]
a = [date[:-6] for date in a]
date_time = pd.to_datetime(a)

df.loc[:,['datetime']] = date_time
df = df.set_index('datetime') 

# ACF and PACF plots
plot_acf(df.precio_spot, lags = 500)
plot_pacf(df.precio_spot, lags = 500)



#############################################
# X_train
timestamp_s = date_time.map(pd.Timestamp.timestamp)
day = 24*60*60
week = 7*day

day_sin = np.array(np.sin(timestamp_s * (2 * np.pi / day)))
day_cos = np.array(np.cos(timestamp_s * (2 * np.pi / day)))
week_sin = np.array(np.sin(timestamp_s * (2 * np.pi / week)))
week_cos = np.array(np.cos(timestamp_s * (2 * np.pi / week)))



# Regressive predictors

# For non vectorized models

#################### TRAINING

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

train_hours = int(0.7*(len(df)-167)+2)
validation_hours = int(0.15*(len(df)-167))
test_hours = int(0.15*(len(df)-167))

train_hours + validation_hours + test_hours == len(df) - 167

len(df)
for i in range(167, train_hours + 167):
    
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


for i in range(train_hours + 167, train_hours + validation_hours + 167):
    
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


for i in range(train_hours+validation_hours + 167, len(df)):
    
    d_test.append(df.demanda_p48[i])
    e_test.append(df.eolica_p48[i])
    
    d_t1_test.append(df.demanda_p48[i-1])
    e_t1_test.append(df.eolica_p48[i-1])
    
    d_t2_test.append(df.demanda_p48[i-2])
    e_t2_test.append(df.eolica_p48[i-2])
    
    d_t3_test.append(df.demanda_p48[i-3])
    e_t3_test.append(df.eolica_p48[i-3])    
    
    d_t24_test.append(df.demanda_p48[i-24])
    e_t24_test.append(df.eolica_p48[i-24])  
    
    d_t168_test.append(df.demanda_p48[i-168])
    e_t168_test.append(df.eolica_p48[i-168])
    
    
d_test = np.array(d_test)
e_test = np.array(e_test)
d_t1_test = np.array(d_t1_test)
e_t1_test = np.array(e_t1_test)
d_t2_test = np.array(d_t2_test)
e_t2_test = np.array(e_t2_test)
d_t3_test = np.array(d_t3_test)
e_t3_test = np.array(e_t3_test)
d_t24_test = np.array(d_t24_test)
e_t24_test = np.array(e_t24_test)
d_t168_test = np.array(d_t168_test)
e_t168_test = np.array(e_t168_test)

# DAY AND WEEK SINE AND COSINE
# TRAINING
day_sin_train = day_sin[167:train_hours + 167]
day_cos_train = day_cos[167:train_hours + 167]
week_sin_train = week_sin[167:train_hours + 167]
week_cos_train = week_cos[167:train_hours + 167]

# VALIDATION
day_sin_val = day_sin[train_hours + 167 : train_hours + validation_hours + 167]
day_cos_val = day_cos[train_hours + 167 : train_hours + validation_hours + 167]
week_sin_val = week_sin[train_hours + 167 : train_hours + validation_hours + 167]
week_cos_val = week_cos[train_hours + 167 : train_hours + validation_hours + 167]

# TEST
day_sin_test = day_sin[train_hours + 167 : train_hours + validation_hours + 167]
day_cos_test = day_cos[train_hours + 167 : train_hours + validation_hours + 167]
week_sin_test = week_sin[train_hours + 167 : train_hours + validation_hours + 167]
week_cos_test = week_cos[train_hours + 167 : train_hours + validation_hours + 167]




# Defining the exogenous variables and the target 
# We take demand and eolic from that hour, the previous 3 hours, one day before and one week before
X_train = [d, e, d_t1, d_t2, e_t2, d_t3, e_t3, d_t24, e_t24, d_t168, e_t168, day_sin_train, day_cos_train, week_sin_train, week_cos_train]
X_val = [d_val, e_val, d_t1_val, d_t2_val, e_t2_val, d_t3_val, e_t3_val, d_t24_val, e_t24_val, d_t168_val, e_t168_val, day_sin_val, day_cos_val, week_sin_val, week_cos_val]
X_test = [d_test, e_test, d_t1_test, e_t1_test, d_t2_test, e_t2_test, e_t3_test, d_t24_test, e_t24_test, d_t168_test, e_t168_test, day_sin_test, day_cos_test, week_sin_test, week_cos_test]
 
target_train = df.precio_spot[167: train_hours + 167]
target_val = df.precio_spot[train_hours + 167: -test_hours]
target_test = df.precio_spot[-test_hours:]

# Training set
X_train = pd.DataFrame(X_train)
# X_train.info()
y_train = pd.DataFrame(target_train)
# y_train.info()

X_val = pd.DataFrame(X_val)
y_val = pd.DataFrame(target_val)
# y_val.info()

X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(target_test)
# y_test.info()

# MLP
df_mlp = MLPRegressor(hidden_layer_sizes=(7), max_iter=2000,
                               activation='relu',solver='adam', learning_rate='adaptive',
                               learning_rate_init= 0.0001, shuffle = False, alpha = 0.0005, random_state=1)

df_mlp.fit(X_train.transpose(), target_train)

# VALIDATION

dates_val = y_val.index
dates_test = y_test.index

# predictions_val = pd.DataFrame(df_mlp.predict(X_val.transpose()), index = dates_val)
predictions_val = pd.DataFrame(df_mlp.predict(X_val.transpose()), index = dates_val, columns = ['Validation Predictions'])


# PLOTTING VALIDATION SET
fig, ax = plt.subplots(figsize=(9, 4))
plt.xlabel('Date')
plt.ylabel('EP')
plt.title('EPF')
y_train.plot(ax=ax,  color = "black", label='Training')
y_val.plot(ax=ax,  color = "blue", label='Validation')
predictions_val.plot(ax=ax,  color = "green", label='Predictions_val')
ax.legend()

# ACCURACY VALIDATION
from sklearn.metrics import mean_squared_error
RMSE_val = math.sqrt(mean_squared_error(y_val, predictions_val))


# TEST
predictions_test = pd.DataFrame(df_mlp.predict(X_test.transpose()), index = dates_test, columns = ['Test Predictions'])


# PLOTTING TEST SET
fig, ax = plt.subplots()
plt.xlabel('Date')
plt.ylabel('EP')
plt.title('EPF')
y_train.plot(ax=ax,  color = "black", label = 'Training')
y_val.plot(ax=ax,  color = "brown", label = 'Validation')
predictions_val.plot(ax=ax,  color = "green", label = 'Predictions_val')
y_test.plot(ax=ax,  color = "grey", label = 'Test')
predictions_test.plot(ax=ax,  color = "blue", label = 'Predictions_test')
ax.legend(['Training', 'Validation', 'Predictions_val', 'Test', 'Predictions_test'])

# ACCURACY 
from sklearn.metrics import mean_squared_error

RMSE_val = math.sqrt(mean_squared_error(y_val, predictions_val))
RMSE_test = math.sqrt(mean_squared_error(y_test, predictions_test))

print('Validation accuracy (RMSE): ', RMSE_val)
print('Test accuracy (RMSE): ', RMSE_test)

def mean_absolute_percentage_error (y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))

MAPE_val = mean_absolute_percentage_error(y_val, predictions_val)
MAPE_test = mean_absolute_percentage_error(y_test, predictions_test)

print('Validation accuracy (MAPE): ', MAPE_val)
print('Test accuracy (MAPE): ', MAPE_test)


# Residuals analysis

#view model summary
df_mlp.summary()

df_mlp.plot_diagnostics(figsize=(7,5))
plt.show()




df_mlp.get_params()



############## VARMAX

from statsmodels.tsa.statespace.varmax import VARMAX
from random import random

def VARMAX_model(X_train, y_train, X_test, y_test):
    # fit model
    model = VARMAX(y_train, exog=X_train.transpose(), order=(1, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.forecast(steps=len(y_test),exog=test['Exog'])
    res=pd.DataFrame({"Pred1":yhat['Act1'], "Pred2":yhat['Act2'], 
            "Act1":test["Act1"].values, "Act2":test["Act2"].values, "Exog":test["Exog"].values})
    return res

y_train = list(target_train)
df_varmax = VARMAX(y_train, exog=X_train.transpose(), order=(1, 1))

show_graph(df_train, df_ret,"Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)")