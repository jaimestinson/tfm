# UNIVARIATE LSTM MODEL forecasting 1h per step (24h of training)

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
df = df.precio_spot


# ACF and PACF plots
plot_acf(df, lags = 200)
plot_pacf(df, lags = 200)

def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)


WINDOW_SIZE = 24
X1, y1 = df_to_X_y(df, WINDOW_SIZE)
X1.shape, y1.shape

train_hours = int(0.7*(len(df)-WINDOW_SIZE)) 
validation_hours = int(0.15*(len(df)-WINDOW_SIZE))
test_hours = int(0.15*(len(df)-WINDOW_SIZE))

train_hours + validation_hours + test_hours == len(df) - WINDOW_SIZE


X_train1, y_train1 = X1[:train_hours], y1[:train_hours]
X_val1, y_val1 = X1[train_hours:train_hours+validation_hours], y1[train_hours:train_hours+validation_hours]
X_test1, y_test1 = X1[train_hours+validation_hours:], y1[train_hours+validation_hours:]
X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

# LSTM (RNN)
model1 = Sequential()
model1.add(InputLayer((24, 1)))
model1.add(LSTM(70))
# meteeeeer esto
model1.add(Dropout(0.2))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))
# 20,737 parameters
model1.summary()

cp1 = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=20, callbacks=[cp1])

plt.plot(range(len(model1.history.history['loss'])), model1.history.history['loss'])
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.show()
# 20 epochs is fine

from tensorflow.keras.models import load_model
model1 = load_model('model1/')


# TRAINING
train_predictions = model1.predict(X_train1).flatten() # output not between []
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})
train_results

import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'][50:1000], color = 'blue')
plt.plot(train_results['Actuals'][50:1000], color = 'green')

# VALIDATION
val_predictions = model1.predict(X_val1).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val1})
val_results

plt.plot(val_results['Val Predictions'][:1000])
plt.plot(val_results['Actuals'][:1000])

# TEST
test_predictions = model1.predict(X_test1).flatten() 
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test1})
test_results

plt.plot(test_results['Test Predictions'][:1000])
plt.plot(test_results['Actuals'][:1000])

from sklearn.metrics import mean_squared_error as mse

def plot_predictions1(model, X, y, start=0, end=1000):
  predictions = model.predict(X).flatten()
  df = pd.DataFrame(data={'Predictions':predictions, 'Actuals':y})
  plt.plot(df['Predictions'][start:end], color = 'blue')
  plt.plot(df['Actuals'][start:end], color = 'green')
  return df, mse(y, predictions)

plot_predictions1(model1, X_test1, y_test1)

# Evaluating the model
# VALIDATION
rmse_val = np.sqrt(np.mean(((val_results['Val Predictions'] - y_val1)**2)))
rmse_val

# TEST
rmse_test = np.sqrt(np.mean(((test_results['Test Predictions'] - y_test1)**2)))
rmse_test



# RESIDUALS ANALYSIS


# Falta ver si escalamos los datos 








##################################################################################

