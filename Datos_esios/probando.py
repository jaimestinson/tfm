# Probando el código 

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

# Generate data
n = 500
t = np.linspace(0,20.0*np.pi,n)
Y = np.sin(t) # X is already between -1 and 1, scaling normally needed

plt.plot(t,Y)
# Set window of past points for LSTM model
window = 100

# Split 80/20 into train/test data
last = int(n/5.0)
Ytrain = Y[:-last]
Ytest = Y[-last-window:]

# Store window number of points as a sequence
yin = []
next_Y = []
for i in range(window,len(Ytrain)):
    yin.append(Ytrain[i-window:i])
    next_Y.append(Ytrain[i])

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
history = m.fit(yin, next_Y, epochs = 100, batch_size = 50,verbose=0)

plt.figure()
plt.ylabel('loss'); plt.xlabel('epoch')
plt.semilogy(history.history['loss'])

# Store "window" points as a sequence
yin = []
next_Y1 = []
for i in range(window,len(Ytest)):
    yin.append(Ytest[i-window:i])
    next_Y1.append(Ytest[i])

# Reshape data to format for LSTM
yin, next_Y1 = np.array(yin), np.array(next_Y1)
yin = yin.reshape((yin.shape[0], yin.shape[1], 1))

# Predict the next value (1 step ahead)
Y_pred = m.predict(yin)

# Plot prediction vs actual for test data
plt.figure()
plt.plot(Y_pred,':',label='LSTM')
plt.plot(next_Y1,'--',label='Actual')
plt.legend()

# Using predicted values to predict next step
Y_pred = Ytest.copy()
for i in range(window,len(Y_pred)):
    yin = Y_pred[i-window:i].reshape((1, window, 1))
    Y_pred[i] = m.predict(yin)

# Plot prediction vs actual for test data
plt.figure()
plt.plot(Y_pred[window:],':',label='LSTM')
plt.plot(next_Y1,'--',label='Actual')
plt.legend()

# measure accuracy

def measure_rmse(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))

RMSE = measure_rmse(next_Y1, Y_pred[window:])
