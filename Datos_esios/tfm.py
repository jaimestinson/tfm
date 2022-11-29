# print('Hello TFM')

# for i in range(1,10):
#     print(i)

# array_1 = np.array([[1, 2, 3],[4, 5, 6]])

# random_Ar = np.random.randint(1,50,(2,5))

# Age = pd.Series([10,20,40],index=['age1','age2','age3'])

# Age.age3

# Filter_Age = Age[Age<30]

# Age.index

a = 83497.43
a.info()
type(a)
#########################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# demand_data = pd.read_csv('export_DemandaProgramadaP48Total_2022-10-06_18_20.csv', sep = ';', index_col = 'datetime', parse_dates = True)
demand_data = pd.read_csv('Demanda_casera.csv',sep = ';', nrows = 35064)

a = [str(date) for date in demand_data['datetime']]
a = [date[:-6] for date in a]
a = pd.to_datetime(a)

demand_data.loc[:,['datetime']] = a
demand_data = demand_data.set_index('datetime') 
# demand_data['value'] = np.float64(demand_data['value'])
# demand_data['value'] = float(demand_data['value'])

demand_data.pop('name')
demand_data.pop('geoid')
demand_data.pop('geoname')
demand_data.pop('id')

demand_data.index
demand_data.index.freq = 'H'

demand_data.info()
pd.DataFrame.head(demand_data)
pd.DataFrame.tail(demand_data)
# demand_data.plot()

import statsmodels as sm
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import seaborn as sns

TR = demand_data.iloc[:26304]
TS = demand_data.iloc[26304:]

TR.info()
pd.DataFrame.head(TR)
pd.DataFrame.tail(TR)

TS.info()
pd.DataFrame.head(TS)
pd.DataFrame.tail(TS)

# transform training data & save lambda value
fitted_data, fitted_lambda = stats.boxcox(TR['precio_spot'])
 
# creating axes to draw plots
fig, ax = plt.subplots(1, 2)
plt.pause(100)
plt.show()

# plotting the original data(non-normal) and
# fitted data (normal)
sns.displot(TR['precio_spot'], hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2},
            label = "Non-Normal", color ="green", ax = ax[0])
 
sns.displot(fitted_data, hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2},
            label = "Normal", color ="green", ax = ax[1])

from pmdarima import auto_arima
 
auto_arima(TR['precio_spot'], trace = True, m = 24, D = 1).summary()

# Decomp_results = sm.tsa.seasonal_decompose(TR)
Decomp_results = seasonal_decompose(TR, extrapolate_trend='freq')
Decomp_results = seasonal_decompose(TR)


print(Decomp_results.trend)
print(Decomp_results.seasonal)
print(Decomp_results.resid)
print(Decomp_results.observed)

Decomp_results.plot()
plt.plot(Decomp_results)
plt.pause(1000)
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(TR, lags = 50)

plot_pacf(TR, lags = 50)

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

s_model = SARIMAX(TR, order = (3, 1, 3), seasonal_order = (0, 0, 0, 0))
s_model = s_model.fit(disp=0)
prediction = s_model.predict(len(TR), len(TR) + len(TS) - 1, typ ='Levels')

plt.plot(TS, color = 'red', label = 'Actual demand')
plt.plot(prediction, color = 'blue', label = 'Predicted demand')
plt.pause(1000)
plt.legend()
plt.xlabel('Hour')
plt.ylabel('Demand')
plt.show()


#a_model = ARIMA(TR, order = (,,))
#predictor =a_model
#predictor.summary()
#predicted_results = predictor.predict(start = len(TR), end = len(TR) + len(TS) - 1, typ = 'Levels')

import math
from sklearn.metrics import mean_squared_error

#rmse = math.sqrt(mean_squared_error(TS, predicted_results))



########################
#Univariate LTSM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# demand_data = pd.read_csv('export_DemandaProgramadaP48Total_2022-10-06_18_20.csv', sep = ';', index_col = 'datetime', parse_dates = True)
demand_data = pd.read_csv('Demanda_casera.csv',sep = ';', nrows = 35064)

a = [str(date) for date in demand_data['datetime']]
a = [date[:-6] for date in a]
a = pd.to_datetime(a)

demand_data.loc[:,['datetime']] = a
demand_data = demand_data.set_index('datetime') 

demand_data.index
demand_data.index.freq = 'H'

TR = demand_data.iloc[:26304].values
TS = demand_data.iloc[26304:27304].values

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

TR_scaled = sc.fit_transform(TR)
TS_scaled = sc.fit_transform(TS)

x_train = []
y_train =[]
WS = 24 #window size

for i in range(WS, len(TR_scaled)):
    x_train.append(TR_scaled[i - WS: i, 0])
    y_train.append(TR_scaled[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

#Developing LSTM model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

model_p = Sequential()

model_p.add(LSTM(units = 60, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model_p.add(Dropout(0.2))

model_p.add(LSTM(units = 60, return_sequences = True))
model_p.add(Dropout(0.2))

model_p.add(LSTM(units = 60, return_sequences = True))
model_p.add(Dropout(0.2))

model_p.add(LSTM(units = 60))
model_p.add(Dropout(0.2))

model_p.add(Dense(units = 1))

model_p.compile(optimizer = 'adam', loss = 'mean_squared_error')

model_p.fit(x_train, y_train, epochs = 15, batch_size = 32)

model_p.save('LSTM - Univariate')

from tensorflow.keras.models import load_model

plt.plot(range(len(model_p.history.history['loss'])), model_p.history.history['loss'])
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.pause(1000)
plt.show()

prediction_test = []

batch_one = TR_scaled[-WS:]
batch_new = batch_one.reshape((1, WS, 1))

for i in range(len(TS)):
    first_pred = model_p.predict(batch_new)[0]
    prediction_test.append(first_pred)
    batch_new = np.append(batch_new[:, 1:, :], [[first_pred]], axis = 1)
    
prediction_test = np.array(prediction_test)

predictions = sc.inverse_transform(prediction_test)

plt.plot(TS, color = 'red', label = 'Actual Values')
plt.plot(predictions, color = 'blue', label = 'predicted Values')
plt.title('LSTM - Univariate Forecast')
plt.xlabel('Time (h)')
plt.ylabel('Demand')
plt.legend()
plt.show()


import math 
from sklearn.metrics import mean_squared_error

RMSE = math.sqrt(mean_squared_error(TS, predictions))

from sklearn.metrics import r2_score

Rsquare = r2_score(TS, predictions)




########################
#Multivariate LTSM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


df = pd.read_csv('Demanda_casera.csv',sep = ';', nrows = 35064)

a = [str(date) for date in df['datetime']]
a = [date[:-6] for date in a]
date_time = pd.to_datetime(a)

df.loc[:,['datetime']] = a
df = df.set_index('datetime') 

timestamp_s = date_time.map(pd.Timestamp.timestamp)
day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

# plt.plot(np.array(df['Day sin'])[:25])
# plt.plot(np.array(df['Day cos'])[:25])
# plt.xlabel('Time [h]')
# plt.title('Time of day signal')

# import seaborn as sn
# sn.heatmap(df.corr())

predictors = ['demand_value', 'eolica', 'Day sin', 'Day cos', 'Year sin', 'Year cos']
target = ['precio_spot']

data_x = pd.get_dummies(df[predictors])
data_y = df.precio_spot

# X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2)
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

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_val = pd.DataFrame(X_val)
y_val = pd.DataFrame(y_val)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

from sklearn.preprocessing import MinMaxScaler

# sc = MinMaxScaler(feature_range = (0,1))
scaler = MinMaxScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)
y_train_scaled = pd.DataFrame(scaler.fit_transform(y_train.values), columns=y_train.columns, index=y_train.index)

X_val_scaled = pd.DataFrame(scaler.fit_transform(X_val.values), columns=X_val.columns, index=X_val.index)
y_val_scaled = pd.DataFrame(scaler.fit_transform(y_val.values), columns=y_val.columns, index=y_val.index)

X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test.values), columns=X_test.columns, index=X_test.index)
y_test_scaled = pd.DataFrame(scaler.fit_transform(y_test.values), columns=y_test.columns, index=y_test.index)


x_train = []
y_train = []
WS = 24 #window size

for i in range(WS, len(X_train_scaled)):
    x_train.append(X_train_scaled[i - WS: i, :])
    y_train.append(y_train_scaled[i, :])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 3)) 

#Developing LSTM model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

model_p = Sequential()

model_p.add(LSTM(units = 70, return_sequences = True, input_shape = (X_train.shape[1], 3)))
model_p.add(Dropout(0.2))

model_p.add(LSTM(units = 70, return_sequences = True))
model_p.add(Dropout(0.2))

model_p.add(LSTM(units = 70, return_sequences = True))
model_p.add(Dropout(0.2))

model_p.add(LSTM(units = 70))
model_p.add(Dropout(0.2))

model_p.add(Dense(units = 1))

model_p.compile(optimizer = 'adam', loss = 'mean_squared_error')

model_p.fit(x_train, y_train, epochs = 15, batch_size = 32)
model_p.fit(x_train, y_train, epochs = 15, batch_size = 32)


model_p.save('LSTM - Multivariate')

from tensorflow.keras.models import load_model

plt.plot(range(len(model_p.history.history['loss'])), model_p.history.history['loss'])
plt.xlabel('Epoch number')
plt.ylabel('Loss')
# plt.pause(1000)
plt.show()

prediction_test = []

batch_one = TR_scaled[-WS:]
batch_new = batch_one.reshape((1, WS, 3))

for i in range(len(TS)): #es el len(TS)
    first_pred = model_p.predict(batch_new)[0]
    prediction_test.append(first_pred)
    new_var = TS_scaled[i, :] 
    new_var = new_var.reshape(1, 2)
    new_test = np.insert(new_var, 2, [first_pred], axis = 1)
    new_test = new_test.reshape(1, 1, 3)
    batch_new = np.append(batch_new[:, 1:, :], new_test, axis = 1)
    
prediction_test = np.array(prediction_test)


SI = MinMaxScaler(feature_range = (0,1))
y_scale = TR[:, 2:3]
SI.fit_transform(y_scale)


predictions = SI.inverse_transform(prediction_test)

real_values = TS[:, 2]

plt.plot(real_values, color = 'red', label = 'Actual Electricity prices')
plt.plot(predictions, color = 'blue', label = 'Predicted Values')
plt.title('LSTM - Multivariate Forecast')
plt.xlabel('Time (h)')
plt.ylabel('Demand')
plt.legend()
plt.show()


import math 
from sklearn.metrics import mean_squared_error

RMSE = math.sqrt(mean_squared_error(TS, predictions))

def mean_absolute_percentage_error (y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))

MAPE = mean_absolute_percentage_error(real_values, predictions)






















