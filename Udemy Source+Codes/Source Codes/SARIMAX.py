"""
SARIMAX

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Temp_Data.csv', index_col = 'DATE', parse_dates = True)
df.info()
pd.DataFrame.head(df)
df.index

df.index.freq = 'D'
df.dropna(inplace=True)

import seaborn as sn
sn.heatmap(df.corr())

train = df.iloc[:510, 0]
test = df.iloc[510:, 0]

exo = df.iloc[:,1:4]
exo_train = exo.iloc[:510]
exo_test = exo.iloc[510:]

from statsmodels.tsa.seasonal import seasonal_decompose

Decomp_results = seasonal_decompose(df['Temp'])

Decomp_results.plot()

from pmdarima import auto_arima

auto_arima(df['Temp'], exogenous = exo, m = 7, trace = True, D=1).summary()


from statsmodels.tsa.statespace.sarimax import SARIMAX

Model = SARIMAX(train, exog = exo_train, order = (2,0,2), seasonal_order = (0,1,1,7) )

Model = Model.fit()

prediction = Model.predict(len(train), len(train)+len(test)-1, exog = exo_test, typ = 'levels' )

plt.plot(test, color = 'red', label = 'Actual Temp')
plt.plot(prediction, color = 'blue', label = 'Predicted Temp')
plt.xlabel('Day')
plt.ylabel('Temp')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test,prediction))


1/5.6