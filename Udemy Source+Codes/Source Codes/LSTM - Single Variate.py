"""
LSTM - Single Variate

"""
# Preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Solar Data Set.csv')
df.dropna(inplace=True)


training_set = df.iloc[:8712,1:2].values
test_set = df.iloc[8712:,1:2].values


from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.fit_transform(test_set)

X_train = []
y_train = []
WS = 24

for i in range(WS, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-WS:i, 0:1])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)
    
# Developing LSTM Model

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

Model_P = Sequential()

Model_P.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1],1)))
Model_P.add(Dropout(0.2))

Model_P.add(LSTM(units = 60, return_sequences = True))
Model_P.add(Dropout(0.2))

Model_P.add(LSTM(units = 60, return_sequences = True))
Model_P.add(Dropout(0.2))

Model_P.add(LSTM(units = 60))
Model_P.add(Dropout(0.2))

Model_P.add(Dense(units = 1))

Model_P.compile(optimizer = 'adam', loss = 'mean_squared_error')

Model_P.fit(X_train, y_train, epochs = 30, batch_size = 32)

Model_P.save('LSTM - UniVariat')

from keras.models import load_model
Model_P = load_model('LSTM - UniVariat')

plt.plot(range(len(Model_P.history.history['loss'])),Model_P.history.history['loss'] )
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()


prediction_test = []

Batch_one = training_set_scaled[-WS:]
Batch_New = Batch_one.reshape((1,WS,1))

for i in range(48):
    
    First_Pred = Model_P.predict(Batch_New)[0]
    
    prediction_test.append(First_Pred)
    
    Batch_New = np.append(Batch_New[:,1:,:], [[First_Pred]], axis = 1)
    

prediction_test = np.array(prediction_test)

predictions = sc.inverse_transform(prediction_test)


plt.plot(test_set, color = 'red', label = 'Actual Values')
plt.plot(predictions, color = 'blue', label = 'predicted Values')
plt.title('LSTM - Univariate Forecast')
plt.xlabel('Time (h)')
plt.ylabel('Solar Irradiance')
plt.legend()
plt.show()

import math

from sklearn.metrics import mean_squared_error

RMSE = math.sqrt(mean_squared_error(test_set, predictions))

from sklearn.metrics import r2_score

Rsquare = r2_score(test_set, predictions)








































