"""
ARIMA

"""
# Import Basic Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('Temp_Data.csv', index_col = 'DATE', parse_dates = True)

df.index.freq = 'D'

df.dropna(inplace = True)

df = pd.DataFrame(df['Temp'])

df.plot()


train = df.iloc[:510,0]
test = df.iloc[510:,0]

# Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

Decomp_results = seasonal_decompose(df)

Decomp_results.plot()

Decomp_results.seasonal.plot()

# Finding the Parameters (p,d,q)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(train, lags=50)

plot_pacf(train, lags=50)


from pmdarima import auto_arima

auto_arima(df, trace = True)


# Developing ARIMA Model

from statsmodels.tsa.arima_model import ARIMA

A_Model = ARIMA(train, order = (1,1,2))

predictor = A_Model.fit()

predictor.summary()

Predicted_results = predictor.predict(start = len(train), end = len(train)+len(test) - 1, typ = 'levels')

plt.plot(test, color = 'red', label = 'Actual Temp')
plt.plot(Predicted_results, color = 'blue', label = 'Predicted Temp')
plt.xlabel ('Day')
plt.ylabel('Temp')
plt.legend()
plt.show()

test.mean()

Predicted_results.mean()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test,Predicted_results))




