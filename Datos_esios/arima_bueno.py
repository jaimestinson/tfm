# Best ARIMA model = (2, 1, 4)

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
from itertools import product
import warnings
warnings.filterwarnings('ignore')

import itertools
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib
import pmdarima as pm
import math

# Use it to decrease memory use

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

def obtain_adf_kpss_results(timeseries, max_d):
    """ Build dataframe with ADF statistics and p-value for time series after applying difference on time series
    
    Args:
        time_series (df): Dataframe of univariate time series  
        max_d (int): Max value of how many times apply difference
        
    Return:
        Dataframe showing values of ADF statistics and p when applying ADF test after applying d times 
        differencing on a time-series.
    
    """
    
    results=[]

    for idx in range(max_d):
        adf_result = adfuller(timeseries, autolag='AIC')
        kpss_result = kpss(timeseries, regression='c', nlags="auto")
        timeseries = timeseries.diff().dropna()
        if adf_result[1] <=0.05:
            adf_stationary = True
        else:
            adf_stationary = False
        if kpss_result[1] <=0.05:
            kpss_stationary = False
        else:
            kpss_stationary = True
            
        stationary = adf_stationary & kpss_stationary
            
        results.append((idx,adf_result[1], kpss_result[1],adf_stationary,kpss_stationary, stationary))
    
    # Construct DataFrame 
    results_df = pd.DataFrame(results, columns=['d','adf_stats','p-value', 'is_adf_stationary','is_kpss_stationary','is_stationary' ])
    
    return results_df


# %matplotlib inline

df = pd.read_csv('Demanda.csv',sep = ';', nrows = 17544)
df = pd.read_csv('Demanda.csv',sep = ';', nrows = 1000)

df.info()
reduce_mem_usage(df)
df.info()

a = [str(date) for date in df['datetime']]
a = [date[:-6] for date in a]
date_time = pd.to_datetime(a)

df.loc[:,['datetime']] = date_time
df = df.set_index('datetime') 

# Box-Jenkins Method
# ad_fuller_result = adfuller(df['precio_spot'])
# print(f'ADF Statistic: {ad_fuller_result[0]}')
# print(f'p-value: {ad_fuller_result[1]}')
# p-value is 2.019150432616161e-15 so we can assume that the trend is stationary


# Seasonal decompose
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

Decomp_results = sm.tsa.seasonal_decompose(df.precio_spot, 
                                          model = 'additive',
                                          period=24, # cycle repeats 8760 hours, i.e., every year
                                         ) 

trend_estimate = Decomp_results.trend
seasonal_estimate = Decomp_results.seasonal
residual_estimate = Decomp_results.resid

# Plotting the time series and it's components together
fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
fig.set_figheight(10)
fig.set_figwidth(20)
# First plot to the Original time series
axes[0].plot(df.precio_spot, label='Original') 
axes[0].legend(loc='upper left');
# second plot to be for trend
axes[1].plot(trend_estimate, label='Trend')
axes[1].legend(loc='upper left');
# third plot to be Seasonality component
axes[2].plot(seasonal_estimate, label='Seasonality')
axes[2].legend(loc='upper left');
# last last plot to be Residual component
axes[3].plot(residual_estimate, label='Residuals')
axes[3].legend(loc='upper left');


# Apply Stationarity Tests
a = obtain_adf_kpss_results(df.precio_spot,5)

def adf_test(sales):
    result=adfuller(sales)
    labels = ['Test parameters', 'p-value','#Lags Used','Dataset observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("Dataset is stationary")
    else:
        print("Dataset is non-stationary ")
        
adf_test(df['precio_spot'])
# este dice que d=0
# d=1, applying differencing only once is enough to make our time series stationary.

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF and PACF plots
# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of df_store_2_item_28_timeon ax1
plot_acf(df.precio_spot,lags=500, zero=False, ax=ax1)

# Plot the PACF of df_store_2_item_28_timeon ax2
plot_pacf(df.precio_spot,lags=500, zero=False, ax=ax2)

plt.show()

# Lags in PACF that are outside confidence interval before having one inside (not counting lag 0)
# best p=8 (AR term)
# best d=1 (diff to make it stationary)
# best q=1

# Seasonal
# best P = 1
# best D = 0
# best Q = 2
# S = 24


# Applying d=1
# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of df_store_2_item_28_timeon ax1
plot_acf(df.precio_spot.diff().dropna(),lags=150, zero=False, ax=ax1)

# Plot the PACF of df_store_2_item_28_timeon ax2
plot_pacf(df.precio_spot.diff().dropna(),lags=150, zero=False, ax=ax2)

plt.show()


# Applying d=2
# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of df_store_2_item_28_timeon ax1
plot_acf(df.precio_spot.diff().diff().dropna(),lags=50, zero=False, ax=ax1)

# Plot the PACF of df_store_2_item_28_timeon ax2
plot_pacf(df.precio_spot.diff().diff().dropna(),lags=50, zero=False, ax=ax2)

plt.show()



# Applying d=3
# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of df_store_2_item_28_timeon ax1
plot_acf(df.precio_spot.diff().diff().diff().dropna(),lags=50, zero=False, ax=ax1)

# Plot the PACF of df_store_2_item_28_timeon ax2
plot_pacf(df.precio_spot.diff().diff().diff().dropna(),lags=50, zero=False, ax=ax2)

plt.show()


#############################################
# Regressive predictors

# For non vectorized models

#################### TRAINING

d = []
e = []
d_t1 = []
e_t1 = []
d_t2 = []
e_t2 = []
d_t3 = []
e_t3 = []
d_t24 = []
e_t24 = []
d_t168 = []
e_t168 = []

train_hours = int(0.7*(len(df)-167)+2)
validation_hours = int(0.15*(len(df)-167))
test_hours = int(0.15*(len(df)-167))

train_hours + validation_hours + test_hours == len(df) - 167

len(df)
for i in range(167, train_hours + 167):
    
    d.append(df.demanda_p48[i])
    e.append(df.eolica_p48[i])
    
    d_t1.append(df.demanda_p48[i-1])
    e_t1.append(df.eolica_p48[i-1])
    
    d_t2.append(df.demanda_p48[i-2])
    e_t2.append(df.eolica_p48[i-2])
    
    d_t3.append(df.demanda_p48[i-3])
    e_t3.append(df.eolica_p48[i-3])    
    
    d_t24.append(df.demanda_p48[i-24])
    e_t24.append(df.eolica_p48[i-24])  
    
    d_t168.append(df.demanda_p48[i-168])
    e_t168.append(df.eolica_p48[i-168])
    
    
d = np.array(d)
e = np.array(e)
d_t1 = np.array(d_t1)
e_t1 = np.array(e_t1)
d_t2 = np.array(d_t2)
e_t2 = np.array(e_t2)
d_t3 = np.array(d_t3)
e_t3 = np.array(e_t3)
d_t24 = np.array(d_t24)
e_t24 = np.array(e_t24)
d_t168 = np.array(d_t168)
e_t168 = np.array(e_t168)


########### VALIDATION

d_val = []
e_val = []
d_t1_val = []
e_t1_val = []
d_t2_val = []
e_t2_val = []
d_t3_val = []
e_t3_val = []
d_t24_val = []
e_t24_val = []
d_t168_val = []
e_t168_val = []


for i in range(train_hours + 167, train_hours + validation_hours + 167):
    
    d_val.append(df.demanda_p48[i])
    e_val.append(df.eolica_p48[i])
    
    d_t1_val.append(df.demanda_p48[i-1])
    e_t1_val.append(df.eolica_p48[i-1])
    
    d_t2_val.append(df.demanda_p48[i-2])
    e_t2_val.append(df.eolica_p48[i-2])
    
    d_t3_val.append(df.demanda_p48[i-3])
    e_t3_val.append(df.eolica_p48[i-3])    
    
    d_t24_val.append(df.demanda_p48[i-24])
    e_t24_val.append(df.eolica_p48[i-24])  
    
    d_t168_val.append(df.demanda_p48[i-168])
    e_t168_val.append(df.eolica_p48[i-168])
    
    
d_val = np.array(d_val)
e_val = np.array(e_val)
d_t1_val = np.array(d_t1_val)
e_t1_val = np.array(e_t1_val)
d_t2_val = np.array(d_t2_val)
e_t2_val = np.array(e_t2_val)
d_t3_val = np.array(d_t3_val)
e_t3_val = np.array(e_t3_val)
d_t24_val = np.array(d_t24_val)
e_t24_val = np.array(e_t24_val)
d_t168_val = np.array(d_t168_val)
e_t168_val = np.array(e_t168_val)


########### TEST


d_test = []
e_test = []
d_t1_test = []
e_t1_test = []
d_t2_test = []
e_t2_test = []
d_t3_test = []
e_t3_test = []
d_t24_test = []
e_t24_test = []
d_t168_test = []
e_t168_test = []


for i in range(train_hours+validation_hours + 167, len(df)):
    
    d_test.append(df.demanda_p48[i])
    e_test.append(df.eolica_p48[i])
    
    d_t1_test.append(df.demanda_p48[i-1])
    e_t1_test.append(df.eolica_p48[i-1])
    
    d_t2_test.append(df.demanda_p48[i-2])
    e_t2_test.append(df.eolica_p48[i-2])
    
    d_t3_test.append(df.demanda_p48[i-3])
    e_t3_test.append(df.eolica_p48[i-3])    
    
    d_t24_test.append(df.demanda_p48[i-24])
    e_t24_test.append(df.eolica_p48[i-24])  
    
    d_t168_test.append(df.demanda_p48[i-168])
    e_t168_test.append(df.eolica_p48[i-168])
    
    
d_test = np.array(d_test)
e_test = np.array(e_test)
d_t1_test = np.array(d_t1_test)
e_t1_test = np.array(e_t1_test)
d_t2_test = np.array(d_t2_test)
e_t2_test = np.array(e_t2_test)
d_t3_test = np.array(d_t3_test)
e_t3_test = np.array(e_t3_test)
d_t24_test = np.array(d_t24_test)
e_t24_test = np.array(e_t24_test)
d_t168_test = np.array(d_t168_test)
e_t168_test = np.array(e_t168_test)



# Defining the exogenous variables and the target 
# We take demand and eolic from that hour, the previous 3 hours, one day before and one week before
X_train = [d, e, d_t1, d_t2, e_t2, d_t3, e_t3, d_t24, e_t24, d_t168, e_t168]
X_val = [d_val, e_val, d_t1_val, d_t2_val, e_t2_val, d_t3_val, e_t3_val, d_t24_val, e_t24_val, d_t168_val, e_t168_val]
X_test = [d_test, e_test, d_t1_test, e_t1_test, d_t2_test, e_t2_test, e_t3_test, d_t24_test, e_t24_test, d_t168_test, e_t168_test]
 
target_train = df.precio_spot[167: train_hours + 167]
target_val = df.precio_spot[train_hours + 167: -test_hours]
target_test = df.precio_spot[-test_hours:]

# Training set
X_train = pd.DataFrame(X_train)
X_train.info()
y_train = pd.DataFrame(target_train)
y_train.info()

X_val = pd.DataFrame(X_val)
y_val = pd.DataFrame(target_val)
y_val.info()

X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(target_test)
y_test.info()






# We can see the Daily seaonality
# Plot
fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(y_train[:], label='Original Series')
axes[0].plot(y_train[:].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)


# Seasinal Dei
axes[1].plot(y_train[:], label='Original Series')
axes[1].plot(y_train[:].diff(24), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('Electricity price', fontsize=16)
plt.show()


# Estimating Coefficients p, q 

from statsmodels.tsa.statespace.sarimax import SARIMAX

# just an example
model = SARIMAX(df['precio_spot'], order=(1,1,1), seasonal_order=(1,1,0,24), freq='H')
results = model.fit(disp=0)
# statistics of the model
results.summary()     


model_1 = SARIMAX(y_train, exog = X_train.transpose(), order=(8,1,1), seasonal_order=(1,0,2,24), freq='H')
results_1 = model_1.fit(disp=0)
# statistics of the model
results_bueno.summary()   

model_2 = SARIMAX(y_train, order=(8,1,1), seasonal_order=(1,0,2,24))
results_2 = model_2.fit(disp=0)
# statistics of the model
results_2.summary()   

# ABRIR R Y VER SI RENTAN LOS RESIDUOS DEL MODELO



# Residuals analysis

results.plot_diagnostics(figsize=(7,5))
plt.show()

results_bueno.plot_diagnostics(figsize=(7,5))
plt.show()

results_2.plot_diagnostics(figsize=(7,5))
plt.show()






# Create empty list to store search results
order_aic_bic=[]

# Loop over p values from 0-6
for p in range(7):
  # Loop over q values from 0-6
    for q in range(7):
      	# create and fit ARMA(p,q) model
        model = SARIMAX(df.precio_spot, order=(p,3,q), freq="H") #because adf test showed that d=1
        results = model.fit()
        
        # Append order and results tuple
        order_aic_bic.append((p,q,results.aic, results.bic))


# Construct DataFrame from order_aic_bic
order_df = pd.DataFrame(order_aic_bic, 
                        columns=['p','q','AIC','BIC'])

# Print order_df in order of increasing AIC
order_df.sort_values('AIC')


# Print order_df in order of increasing BIC
order_df.sort_values('BIC')


# Best model (4,1,6)
# Best model (3,2,6)
# Best model (6,3,3)


arima_model_1 = SARIMAX(df.precio_spot, order=(4,1,6))
# fit model
arima_results_1 = arima_model_1.fit()

# Calculate the mean absolute error from residuals
mae_1 = np.mean(np.abs(arima_results_1.resid))

# Print mean absolute error
print('MAE_1: %.3f' % mae_1)


# Diagnostic Summary Statistics
arima_results_1.summary()

# Plot Diagnostics
# Create the 4 diagostics plots using plot_diagnostics method
arima_results_1.plot_diagnostics()
plt.show()

###########################################################
arima_model_2 = SARIMAX(df.precio_spot, order=(3,2,6))
# fit model
arima_results_2 = arima_model_2.fit()

# Calculate the mean absolute error from residuals
mae_2 = np.mean(np.abs(arima_results_2.resid))

# Print mean absolute error
print('MAE_2: %.3f' % mae_2)


# Diagnostic Summary Statistics
arima_results_2.summary()

# Plot Diagnostics
# Create the 4 diagostics plots using plot_diagnostics method
arima_results_2.plot_diagnostics()
plt.show()

##############################################################
arima_model_3 = SARIMAX(df.precio_spot, order=(6,3,3))
# fit model
arima_results_3 = arima_model_3.fit()

# Calculate the mean absolute error from residuals
mae_3 = np.mean(np.abs(arima_results_3.resid))

# Print mean absolute error
print('MAE_3: %.3f' % mae_3)


# Diagnostic Summary Statistics
arima_results_3.summary()

# Plot Diagnostics
# Create the 4 diagostics plots using plot_diagnostics method
arima_results_3.plot_diagnostics()
plt.show()

############ comparison


print('MAE_1: %.3f' % mae_1)
print('MAE_2: %.3f' % mae_2)
print('MAE_3: %.3f' % mae_3)















# Defining the exogenous variables and the target 
predictors = ['demanda_p48', 'eolica_p48']
target = 'precio_spot'

data_x = pd.get_dummies(df[predictors])
data_y = df.precio_spot

n = len(df)

# Training set
X_train = data_x[0:int(n*0.7)]
y_train = data_y[0:int(n*0.7)]

# # Validation set
# X_val = data_x[int(n*0.7):int(n*0.85)]
# y_val = data_y[int(n*0.7):int(n*0.85)]

# Test set
X_test = data_x[int(n*0.7):]
y_test = data_y[int(n*0.7):]

y_train.astype('float32')


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

s_model = SARIMAX(y_train, order = (5, 1, 1), seasonal_order = (0, 0, 1, 24), trend = 'ct')
s_model = s_model.fit(disp=0)
prediction = s_model.predict(len(X_train), len(X_train) + len(X_test) - 1, typ ='Levels')

fig, ax = plt.subplots(1, 1)
ax.set_title("Residuals")
ax.plot(s_model.resid, label="Python", linestyle=":", alpha=.8)


plt.plot(y_test, color = 'red', label = 'Actual demand')
plt.plot(prediction, color = 'blue', label = 'Predicted spot price')
plt.pause(1000)
plt.legend()
plt.xlabel('Hour')
plt.ylabel('Spot price')
plt.show()

plt.figure()
plt.plot(prediction, color='red',label='MLP-Prediction')
plt.plot(y_test,color='blue',label='Actual')
plt.legend()
