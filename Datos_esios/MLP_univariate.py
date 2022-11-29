# MLP univariate LSTM-like (1h per step)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv',sep = ';', nrows = 17544)

a = [str(date) for date in df['datetime']]
a = [date[:-6] for date in a]
date_time = pd.to_datetime(a)

df.loc[:,['datetime']] = date_time
df = df.set_index('datetime') 
df = df[['precio_spot']]

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
scaled_set = sc.fit_transform(df[['precio_spot']])

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []

train_hours = int(0.7*(len(df))+2)
validation_hours = int(0.15*(len(df)))
test_hours = int(0.15*(len(df)))

train_hours + validation_hours + test_hours == len(df) 
n_steps = 48

# Training set
for i in range(n_steps, train_hours): # len(X_train) = 1258 - 60 = 1198
    X_train.append(scaled_set[i-n_steps:i, 0])
    y_train.append(scaled_set[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

len(X_train)

# Validation set
for i in range(train_hours, train_hours + validation_hours): # len(X_train) = 1258 - 60 = 1198
    X_val.append(scaled_set[i-n_steps:i, 0])
    y_val.append(scaled_set[i, 0])
X_val, y_val = np.array(X_val), np.array(y_val)

len(X_val)

# Test set
for i in range(train_hours + validation_hours, len(df)): # len(X_train) = 1258 - 60 = 1198
    X_test.append(scaled_set[i-n_steps:i, 0])
    y_test.append(scaled_set[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)


len(X_test)

# Reshaping
X_train.shape
y_train.shape

X_val.shape
y_val.shape

X_test.shape
y_test.shape

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1))

X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
y_val = np.reshape(y_val, (y_val.shape[0], 1))

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


from tensorflow.keras.callbacks import ModelCheckpoint


# Initializing the RNN
mlp_model = Sequential()

# Adding first MLP layer and some Dropout regularisation
mlp_model.add(Dense(50, activation='relu', input_dim=n_steps))
# Adding an output layer
mlp_model.add(Dense(units=1))

mlp_model.summary()
# 71,051 parameters

# Compiling the RNN
mlp_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
cp1 = ModelCheckpoint('mlp_model/', save_best_only=True)

# Fitting the RNN to the training set
mlp_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs = 20, batch_size = 32, callbacks=[cp1]) # Batch size to be loaded in the RAM indirectly 

plt.plot(range(len(mlp_model.history.history['loss'])), mlp_model.history.history['loss'])
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



test_predictions = mlp_model.predict(X_test)
test_predictions = sc.inverse_transform(test_predictions)


# TRAINING
train_predictions = []
train_predictions = mlp_model.predict(X_train) # output not between []
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
val_predictions = mlp_model.predict(X_val)
val_predictions = sc.inverse_transform(val_predictions)
# val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val})
# val_results

# # Plotting validation set
# plt.plot(val_results['Val Predictions'])
# plt.plot(val_results['Actuals'])

plt.plot(y_train, color = 'blue')
plt.plot(val_predictions, color = 'green')


# TEST
test_predictions = []
test_predictions = mlp_model.predict(X_test)
test_predictions = sc.inverse_transform(test_predictions)
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
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

mlp_model.weights
