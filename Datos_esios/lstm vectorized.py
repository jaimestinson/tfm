# UNIVARIATE LSTM MODEL forecasting 24h per step (7 days of training)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

# from keras import metrics
# import statsmodels.api as sm

# Reading the data
df = pd.read_csv('dataset.csv',sep = ';', nrows = 17544)

# Preprocessing the data
a = [str(date) for date in df['datetime']]
a = [date[:-6] for date in a]
date_time = pd.to_datetime(a)
df.loc[:,['datetime']] = date_time
df = df.set_index('datetime') 
df = df.precio_spot


# Defining df as a series separated in days in which there is a 24h item vector

days = int(len(df)/24)
# df_days = []

# for i in range(days):
#     df_days.append(df[i*24:(i+1)*24])
    
    
# WINDOW_SIZE = 168 #h
 
# train_hours = int(0.7*(len(df)-WINDOW_SIZE)) + 1
# validation_hours = int(0.15*(len(df)-WINDOW_SIZE))
# test_hours = int(0.15*(len(df)-WINDOW_SIZE))

# train_hours + validation_hours + test_hours == len(df) - WINDOW_SIZE

days = int(len(df)/24)

def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)

  return np.array(X)


WINDOW_DAYS = 7
train_days = int(0.7*(days-WINDOW_DAYS)) + 2
validation_days = int(0.15*(days-WINDOW_DAYS))
test_days = int(0.15*(days-WINDOW_DAYS))

train_days + validation_days + test_days == days - WINDOW_DAYS

X1 = []
y = []
for i in range(days-WINDOW_DAYS):
    row = [a for a in df[(i+WINDOW_DAYS)*24:(i+WINDOW_DAYS+1)*24]]
    y.append(row)
    
y = np.array([np.array(i) for i in y])
# HAY y  UN NP.ARRAY()

for i in range(days-WINDOW_DAYS):
    row = [a for a in df[i*24:(i+WINDOW_DAYS)*24]]
    X1.append(row)
    
X1 = np.array(X1)
y = np.array(y)

X_train1 = X1[:train_days]
y_train1 = y[:train_days]
X_val1, y_val1 = X1[train_days:train_days+validation_days], y[train_days:train_days+validation_days]
X_test1, y_test1 = X1[train_days+validation_days:], y[train_days+validation_days:]
X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape



# X_train es un array cuya fila contiene las 168 observaciones desde el dia i para predecir el dia i+7
# y_train es un array cuya fila contiene las 24 del dia i+7

##################### MODEL DEFINITION

# LSTM (Vector RNN)
model1 = Sequential()
model1.add(InputLayer((WINDOW_DAYS*24, 1)))
model1.add(LSTM(64))
model1.add(Dropout(0.2))
model1.add(Dense(8, 'relu'))
model1.add(Dropout(0.2))
model1.add(Dense(24, 'linear'))
# 17632 parameters
model1.summary()

cp1 = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=500, callbacks=[cp1])


# 
plt.plot(range(len(model1.history.history['loss'])), model1.history.history['loss'])
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.show()


from tensorflow.keras.models import load_model
model1 = load_model('model1/')

train_predictions = []

# TRAINING
train_predictions = model1.predict(X_train1) 
train_predictions = train_predictions.flatten()
y_train1 = y_train1.flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})
train_results.info

# Plotting training set
plt.plot(train_results['Train Predictions'], color = 'blue')
plt.plot(train_results['Actuals'], color = 'green')

# VALIDATION
val_predictions = model1.predict(X_val1).flatten()
y_val1 = y_val1.flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val1})
val_results

# Plotting validation set
plt.plot(val_results['Val Predictions'])
plt.plot(val_results['Actuals'])

# TEST
test_predictions = model1.predict(X_test1).flatten() 
y_test1 = y_test1.flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test1})
test_results

# Plotting test set
plt.plot(test_results['Test Predictions'])
plt.plot(test_results['Actuals'])


# Evaluating the model
# VALIDATION
rmse_val = np.sqrt(np.mean(((val_results['Val Predictions'] - y_val1)**2)))
rmse_val

# TEST
rmse_test = np.sqrt(np.mean(((test_results['Test Predictions'] - y_test1)**2)))
rmse_test




















########################################################

WINDOW_DAYS = 14
train_days = int(0.7*(days-WINDOW_DAYS)) + 2
validation_days = int(0.15*(days-WINDOW_DAYS))
test_days = int(0.15*(days-WINDOW_DAYS))

train_days + validation_days + test_days == days - WINDOW_DAYS

X1 = []
y = []
for i in range(days-WINDOW_DAYS):
    row = [a for a in df[(i+WINDOW_DAYS)*24:(i+WINDOW_DAYS+1)*24]]
    y.append(row)
    
y = np.array([np.array(i) for i in y])
# HAY y  UN NP.ARRAY()

for i in range(days-WINDOW_DAYS):
    row = [a for a in df[i*24:(i+WINDOW_DAYS)*24]]
    X1.append(row)
    
X1 = np.array(X1)
y = np.array(y)

X_train1 = X1[:train_days]
y_train1 = y[:train_days]
X_val1, y_val1 = X1[train_days:train_days+validation_days], y[train_days:train_days+validation_days]
X_test1, y_test1 = X1[train_days+validation_days:], y[train_days+validation_days:]
X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape



# X_train es un array cuya fila contiene las 168 observaciones desde el dia i para predecir el dia i+7
# y_train es un array cuya fila contiene las 24 del dia i+7

##################### MODEL DEFINITION

# LSTM (Vector RNN)
model1 = Sequential()
model1.add(InputLayer((WINDOW_DAYS*24, 1)))
model1.add(LSTM(64))
model1.add(Dropout(0.2))
model1.add(Dense(8, 'relu'))
model1.add(Dropout(0.2))
model1.add(Dense(24, 'linear'))
# 17632 parameters
model1.summary()

cp1 = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=500, callbacks=[cp1])


# 
plt.plot(range(len(model1.history.history['loss'])), model1.history.history['loss'])
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.show()


from tensorflow.keras.models import load_model
model1 = load_model('model1/')

train_predictions = []

# TRAINING
train_predictions = model1.predict(X_train1) 
train_predictions = train_predictions.flatten()
y_train1 = y_train1.flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})
train_results.info

# Plotting training set
plt.plot(train_results['Train Predictions'], color = 'blue')
plt.plot(train_results['Actuals'], color = 'green')

# VALIDATION
val_predictions = model1.predict(X_val1).flatten()
y_val1 = y_val1.flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val1})
val_results

# Plotting validation set
plt.plot(val_results['Val Predictions'])
plt.plot(val_results['Actuals'])

# TEST
test_predictions = model1.predict(X_test1).flatten() 
y_test1 = y_test1.flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test1})
test_results

# Plotting test set
plt.plot(test_results['Test Predictions'])
plt.plot(test_results['Actuals'])


# Evaluating the model
# VALIDATION
rmse_val = np.sqrt(np.mean(((val_results['Val Predictions'] - y_val1)**2)))
rmse_val

# TEST
rmse_test = np.sqrt(np.mean(((test_results['Test Predictions'] - y_test1)**2)))
rmse_test
















