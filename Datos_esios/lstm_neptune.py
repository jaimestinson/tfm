# LSTM (24h per step)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv',sep = ';', nrows = 17544)

a = [str(date) for date in df['datetime']]
a = [date[:-6] for date in a]
date_time = pd.to_datetime(a)
df.loc[:,['datetime']] = date_time
df = df.set_index('datetime') 
df = df.precio_spot
# Feature scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
scaled_set = sc.fit_transform([df])


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

X_train = X1[:train_days]
y_train = y[:train_days]
X_val, y_val = X1[train_days:train_days+validation_days], y[train_days:train_days+validation_days]
X_test, y_test = X1[train_days+validation_days:], y[train_days+validation_days:]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], 24))

X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
y_val = np.reshape(y_val, (y_val.shape[0], 24))

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 24))

# X_train = []
# y_train = []
# X_val = []
# y_val = []
# X_test = []
# y_test = []

# train_hours = int(0.7*(len(df))+2)
# validation_hours = int(0.15*(len(df)))
# test_hours = int(0.15*(len(df)))

# train_hours + validation_hours + test_hours == len(df) 


# n_lags = 24 * 28

# # Training set
# for i in range(n_lags, train_hours): # len(X_train) = 1258 - 60 = 1198
#     X_train.append(scaled_set[i-n_lags:i, 0])
#     y_train.append(scaled_set[i, 0])
# X_train, y_train = np.array(X_train), np.array(y_train)

# # Validation set
# for i in range(train_hours, train_hours + validation_hours): # len(X_train) = 1258 - 60 = 1198
#     X_val.append(scaled_set[i-n_lags:i, 0])
#     y_val.append(scaled_set[i, 0])
# X_val, y_val = np.array(X_val), np.array(y_val)

# # Test set
# for i in range(train_hours + validation_hours, len(df)): # len(X_train) = 1258 - 60 = 1198
#     X_test.append(scaled_set[i-n_lags:i, 0])
#     y_test.append(scaled_set[i, 0])
# X_test, y_test = np.array(X_test), np.array(y_test)

# # X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[2], 1))
# # y_test = np.reshape(y_test, (y_test.shape[1], 1))

# # Reshaping
# X_train.shape
# y_train.shape

# X_val.shape
# y_val.shape

# X_test.shape
# y_test.shape

# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# y_train = np.reshape(y_train, (y_train.shape[0], 1))

# X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
# y_val = np.reshape(y_val, (y_val.shape[0], 1))

# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# y_test = np.reshape(y_test, (y_test.shape[0], 1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Initializing the RNN
regressor = Sequential()

# Adding first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 70, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 70, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 70, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 70, return_sequences = False))
regressor.add(Dropout(0.2))

# Adding an output layer
regressor.add(Dense(units=24))

regressor.summary()
# 71,051 parameters

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
cp1 = ModelCheckpoint('regressor/', save_best_only=True)

# Fitting the RNN to the training set
regressor.fit(X_train, y_train, validation_data=(X_val, y_val), epochs = 150, batch_size = 32, callbacks=[cp1]) # Batch size to be loaded in the RAM indirectly 

plt.plot(range(len(regressor.history.history['loss'])), regressor.history.history['loss'])
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.show()

# y_train[-3:]
# y_val[0:3]
# y_val[-3:]
# y_test[0:3]


# pd.DataFrame(y_train)
# pd.DataFrame(y_val)
# pd.DataFrame(y_test)



# # Reshaping
# X_test = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

# dataset_total = pd.concat((y_train, y_val, y_test), axis = 0)
# dataset_total = y_train + y_val + y_test


y_train = sc.inverse_transform(y_train)
y_val = sc.inverse_transform(y_val)
y_test = sc.inverse_transform(y_test)



test_predictions = regressor.predict(X_test)
test_predictions = sc.inverse_transform(test_predictions)


# TRAINING
train_predictions = []
train_predictions = regressor.predict(X_train) # output not between []
train_predictions = sc.inverse_transform(train_predictions)
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})
len(train_results)

# Plotting training set
plt.plot(train_results['Train Predictions'], color = 'blue')
plt.plot(train_results['Actuals'], color = 'green')

plt.plot(y_train, color = 'blue')
plt.plot(train_predictions, color = 'green')

# VALIDATION
val_predictions = []
val_predictions = regressor.predict(X_val)
val_predictions = sc.inverse_transform(val_predictions)
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val}, index = val_index)
val_results

# Plotting validation set
plt.plot(val_results['Val Predictions'])
plt.plot(val_results['Actuals'])

plt.plot(y_train, color = 'blue')
plt.plot(val_predictions, color = 'green')


# TEST
test_predictions = []
test_predictions = regressor.predict(X_test)
test_predictions = sc.inverse_transform(test_predictions)
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test}, index = test_index)
test_results

# Plotting test set
plt.plot(test_results['Test Predictions'])
plt.plot(test_results['Actuals'])

plt.plot(y_test, color = 'blue')
plt.plot(test_predictions, color = 'green')




# Evaluating the model
# VALIDATION
rmse_val = np.sqrt(np.mean(((val_results['Val Predictions'] - y_val)**2)))
rmse_val

# TEST
rmse_test = np.sqrt(np.mean(((test_predictions - y_test)**2)))
rmse_test




###############################################################################






for i in range(5,200,5):
    # Initializing the RNN
    regressor = Sequential()
    
    # Adding first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = False))
    regressor.add(Dropout(0.2))
    
    # Adding an output layer
    regressor.add(Dense(units=1))
    
    regressor.summary()
    # 71,051 parameters
    
    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    cp1 = ModelCheckpoint('regressor/', save_best_only=True)
    
    # Fitting the RNN to the training set
    regressor.fit(X_train, y_train, validation_data=(X_val, y_val), epochs = 20, batch_size = 32, callbacks=[cp1]) # Batch size to be loaded in the RAM indirectly 


    y_train = sc.inverse_transform(y_train)
    y_val = sc.inverse_transform(y_val)
    y_test = sc.inverse_transform(y_test)
    
    test_predictions = regressor.predict(X_test)
    test_predictions = sc.inverse_transform(test_predictions)
    
    # TEST
    test_predictions = []
    test_predictions = regressor.predict(X_test)
    test_predictions = sc.inverse_transform(test_predictions)
    







