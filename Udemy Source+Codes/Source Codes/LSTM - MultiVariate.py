"""
LSTM - MultiVariate

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# DATA Reference: https://www.ieso.ca/power-data

df = pd.read_csv('Electricity Consumption.csv')
df.dropna(inplace=True)


import seaborn as sn
sn.heatmap(df.corr())

training_set = df.iloc[:8712, 1:4].values
test_set = df.iloc[8712:, 1:4].values

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.fit_transform(test_set)

test_set_scaled = test_set_scaled[:, 0:2]

X_train = []
y_train = []
WS = 24

for i in range(WS, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-WS:i, 0:3])
    y_train.append(training_set_scaled[i,2])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 3))

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


Model = Sequential()

Model.add(LSTM(units = 70, return_sequences = True, input_shape = (X_train.shape[1], 3)))
Model.add(Dropout(0.2))

Model.add(LSTM(units = 70, return_sequences = True))
Model.add(Dropout(0.2))

Model.add(LSTM(units = 70, return_sequences = True))
Model.add(Dropout(0.2))

Model.add(LSTM(units = 70))
Model.add(Dropout(0.2))

Model.add(Dense(units = 1))

Model.compile(optimizer = 'adam', loss = 'mean_squared_error')

Model.fit(X_train,y_train, epochs = 80, batch_size = 32)

plt.plot(range(len(Model.history.history['loss'])), Model.history.history['loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()

Model.save('LSTM - Multivariate')

from keras.models import load_model
Model = load_model('LSTM - Multivariate')

prediction_test = []

Batch_one = training_set_scaled[-24:]
Batch_New = Batch_one.reshape((1,24,3))

for i in range(48):
    
    First_pred = Model.predict(Batch_New)[0]
    
    prediction_test.append(First_pred)
    
    New_var = test_set_scaled[i,:]
    
    New_var = New_var.reshape(1,2)
    
    New_test = np.insert(New_var, 2, [First_pred], axis =1)
    
    New_test = New_test.reshape(1,1,3)
    
    Batch_New = np.append(Batch_New[:,1:,:], New_test, axis=1)
    
prediction_test = np.array(prediction_test)

SI = MinMaxScaler(feature_range = (0,1))
y_Scale = training_set[:,2:3]
SI.fit_transform(y_Scale)

predictions = SI.inverse_transform(prediction_test)

real_values = test_set[:, 2]

plt.plot(real_values, color = 'red', label = 'Actual Electrical Consumption')
plt.plot(predictions, color = 'blue', label = 'Predicted Values')
plt.title('Electrical Consumption Prediction')
plt.xlabel('Time (hr)')
plt.ylabel('Electrical Demand (MW)')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
RMSE = math.sqrt(mean_squared_error(real_values,predictions))


def mean_absolute_percentage_error (y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true)) * 100

MAPE = mean_absolute_percentage_error(real_values,predictions)
















