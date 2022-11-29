import pandas as pd
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib as mpl
import seaborn as sns
import tensorflow as tf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import math



df = pd.read_csv('dataset.csv',sep = ';', nrows = 17545)

a = [str(date) for date in df['datetime']]
a = [date[:-6] for date in a]
date_time = pd.to_datetime(a)

df.loc[:,['datetime']] = date_time
df = df.set_index('datetime') 

timestamp_s = date_time.map(pd.Timestamp.timestamp)
day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

# Correlation between the exogenous variables
sns.heatmap(df.corr())

# ACF and PACF plots
plot_acf(df.precio_spot, lags = 50)
plot_pacf(df.precio_spot, lags = 50)

# Defining the exogenous variables and the target 
predictors = ['demanda_p48', 'eolica_p48', 'Day sin', 'Day cos', 'Year sin', 'Year cos']
target = 'precio_spot'

data_x = pd.get_dummies(df[predictors])
data_y = df.precio_spot

# X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2)
# n = len(df)

# # Training set
# X_train = data_x[0:int(n*0.7)]
# y_train = data_y[0:int(n*0.7)]

# # # Validation set
# # X_val = data_x[int(n*0.7):int(n*0.85)]
# # y_val = data_y[int(n*0.7):int(n*0.85)]

# # Test set
# X_test = data_x[int(n*0.7):]
# y_test = data_y[int(n*0.7):]

# # DataFrame conversion
# X_train = pd.DataFrame(X_train)
# y_train = pd.DataFrame(y_train)
# # X_val = pd.DataFrame(X_val)
# # y_val = pd.DataFrame(y_val)
# X_test = pd.DataFrame(X_test)
# y_test = pd.DataFrame(y_test)


# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#Developing LSTM model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

model_p = Sequential()

model_p.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model_p.add(Dropout(0.2))

model_p.add(LSTM(units = 60, return_sequences = True))
model_p.add(Dropout(0.2))

model_p.add(LSTM(units = 60, return_sequences = True))
model_p.add(Dropout(0.2))

model_p.add(LSTM(units = 60))
model_p.add(Dropout(0.2))

model_p.add(Dense(units = 1))

model_p.compile(optimizer = 'adam', loss = 'mean_squared_error')

model_p.fit(X_train, y_train, epochs = 3, batch_size = 32)

model_p.save('LSTM - Univariate')

from tensorflow.keras.models import load_model

plt.plot(range(len(model_p.history.history['loss'])), model_p.history.history['loss'])
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.pause(1000)
plt.show()

# prediction_test = []

# WS = 24

# batch_one = X_train[-WS:]
# batch_new = batch_one.reshape((1, WS, 1))

# for i in range(len(X_test)):
#     first_pred = model_p.predict(batch_new)[0]
#     prediction_test.append(first_pred)
#     batch_new = np.append(batch_new[:, 1:, :], [[first_pred]], axis = 1)
    
# prediction_test = np.array(prediction_test)

# model = Sequential()

# model.add(Dense(512))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))

# model.compile(optimizer='Adam',
#               loss='rmse',
#               metrics=['accuracy'])

# model.fit(X_train, y_train,
#           batch_size=32,
#           epochs=15,
#           verbose=1,
#           validation_data=(X_test, y_test))

# score = model.evaluate(X_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


##########################################################################################################

n = len(df)
window = 168 # una semana

# Training set 70%
X_train = data_x[0:int(n*0.7)]
y_train = data_y[0:int(n*0.7)]

# Test set
X_test = data_x[int(n*0.7)-window:]
y_test = data_y[int(n*0.7)-window:]



# Store window number of points as a sequence
yin = []
next_Y = []
for i in range(window,len(y_train)):
    yin.append(y_train[i-window:i])
    next_Y.append(y_train[i])

# Reshape data to format for LSTM
yin, next_Y = np.array(yin), np.array(next_Y)
yin = yin.reshape(yin.shape[0], yin.shape[1], 1)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize LSTM model
m = Sequential()
m.add(LSTM(units=50, return_sequences=True, input_shape=(yin.shape[1],1)))
m.add(Dropout(0.2))
m.add(LSTM(units=50))
m.add(Dropout(0.2))
m.add(Dense(units=1))
m.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit LSTM model
history = m.fit(yin, next_Y, epochs = 100, batch_size = 64,verbose=0)

plt.figure()
plt.ylabel('loss'); plt.xlabel('epoch')
plt.semilogy(history.history['loss'])

# Store "window" points as a sequence
# yin = []
# next_Y1 = []
# for i in range(window,len(y_test)):
#     yin.append(y_test[i-window:i])
#     # yin se llena con las #window observaciones previas a la hora i 
#     next_Y1.append(y_test[i])
#     # next_Y1 se llena con todos los precios del periodo test menos el primer dia

# # Reshape data to format for LSTM
# yin, next_Y1 = np.array(yin), np.array(next_Y1)
# yin = yin.reshape((yin.shape[0], yin.shape[1], 1))

# # Predict the next value (1 step ahead)
# Y_pred = m.predict(yin)
# para cada vector de observaciones de ytest se predice una salida
# esa prediccion no entra en el vector siguiente

# Plot prediction vs actual for test data
plt.figure()
plt.plot(Y_pred,':',label='LSTM-Prediction')
plt.plot(next_Y1,'--',label='Actual')
plt.legend()

# Using predicted values to predict next step
Y_pred = y_test.copy()
Y_pred = np.array(Y_pred)
for i in range(window,len(y_test)):
    yin = Y_pred[i-window:i].reshape((1, window, 1))
    Y_pred[i] = m.predict(yin)

################################3






# Plot prediction vs actual for test data
plt.figure()
plt.plot(Y_pred[window:],':',label='LSTM-Prediction')
plt.plot(next_Y1,'--',label='Actual')
plt.legend()

# measure accuracy

def measure_rmse(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))

RMSE = measure_rmse(next_Y1, Y_pred[window:])

