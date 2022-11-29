# MULTIVARIATE VARMAX MODEL forecasting 24h per step (7 days of training)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

from statsmodels.tsa.statespace.varmax import VARMAX

import warnings
warnings.filterwarnings('ignore')

# Importing the dataset

df = pd.read_csv('dataset.csv',sep = ';', nrows = 17544)

a = [str(date) for date in df['datetime']]
a = [date[:-6] for date in a]
date_time = pd.to_datetime(a)

df.loc[:,['datetime']] = date_time
df = df.set_index('datetime') 

# Training, Validation and Test Split

days = int(len(df)/24)

train_days = int(0.7*(days)) + 2
validation_days = int(0.15*(days))
test_days = int(0.15*(days))

train_days + validation_days + test_days - days

# Training set
df_train = df[:train_days]

# Validation set
df_val = df[train_days:train_days+validation_days]

# Test set
df_test = df[train_days+validation_days:]



# Training set
df_train = df[:-24]

# Validation set
df_val = df[train_days:train_days+validation_days]

# Test set
df_test = df[-24:]
######################## MODEL DEFINITION


# Lags in PACF that are outside confidence interval before having one inside (not counting lag 0)
# best p=8 (AR term)
# best d=1 (diff to make it stationary)
# best q=1

# Seasonal
# best P = 1
# best D = 0
# best Q = 2
# S = 24


df_train_diff = df_train.diff()[1:]


model = VAR(df_train_diff)
# sorted_order = model.select_order(maxlags=30)
# print(sorted_order.summary())

var_model = VARMAX(df_train, order=(8,0), enforce_stationarity= True)
fitted_model = var_model.fit()
print(fitted_model.summary())

n_forecast = 24
predict = fitted_model.get_prediction(start=len(df_train), end=len(df_train) + n_forecast - 1)
predictions = predict.predicted_mean


predictions.columns=['demanda_p48_forecast','eolica_p48_forecast','precio_spot_forecast']


test_vs_pred = pd.concat([df_test, predictions],axis=1)

test_vs_pred.plot(figsize=(12,5))




# RESIDUALS ANALYSIS














