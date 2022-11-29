# UNIVARIATE SARIMAX MODEL 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

# Importing the dataset

df = pd.read_csv('dataset.csv',sep = ';', nrows = 17544)

a = [str(date) for date in df['datetime']]
a = [date[:-6] for date in a]
date_time = pd.to_datetime(a)

df.loc[:,['datetime']] = date_time
df = df.set_index('datetime') 

df = df.precio_spot

model = sm.tsa.statespace.SARIMAX(df, order=(8, 1, 1),seasonal_order=(1, 0, 2, 24))
results = model.fit()

df['forecast']=results.predict(start=90,end=103,dynamic=True)
df[['precio_spot','forecast']].plot(figsize=(12,8))

