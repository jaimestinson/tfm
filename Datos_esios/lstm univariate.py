# lstm univariate predicting 1h each step

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import datetime
import math

import IPython
import IPython.display

import matplotlib as mpl
import seaborn as sns
import tensorflow as tf

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
 
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


df = pd.read_csv('dataset.csv',sep = ';', nrows = 17544)

a = [str(date) for date in df['datetime']]
a = [date[:-6] for date in a]
date_time = pd.to_datetime(a)

df.loc[:,['datetime']] = date_time
df = df.set_index('datetime') 
df = df.precio_spot

def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)


WINDOW_SIZE = 24
X1, y1 = df_to_X_y(df, WINDOW_SIZE)
X1.shape, y1.shape

train_hours = int(0.7*(len(df)-WINDOW_SIZE)) 
validation_hours = int(0.15*(len(df)-WINDOW_SIZE))
test_hours = int(0.15*(len(df)-WINDOW_SIZE))

train_hours + validation_hours + test_hours == len(df) - WINDOW_SIZE


X_train1, y_train1 = X1[:train_hours], y1[:train_hours]
X_val1, y_val1 = X1[train_hours:train_hours+validation_hours], y1[train_hours:train_hours+validation_hours]
X_test1, y_test1 = X1[train_hours+validation_hours:], y1[train_hours+validation_hours:]
X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape





















































