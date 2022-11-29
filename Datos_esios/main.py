# MODELO ARIMA SENCILLO

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



import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter
import statsmodels.api as sm
import matplotlib
import pmdarima as pm

# %matplotlib inline

df = pd.read_csv('Demanda.csv',sep = ';', nrows = 35064)
df = pd.read_csv('Demanda.csv',sep = ';', nrows = 17545)
df = pd.read_csv('Demanda.csv',sep = ';', nrows = 8761)


df.tail()

a = [str(date) for date in df['datetime']]
a = [date[:-6] for date in a]
date_time = pd.to_datetime(a)

df.loc[:,['datetime']] = date_time
df = df.set_index('datetime') 

# writer = ExcelWriter('C:/Users/jaime/OneDrive/Desktop/TFM/dataset.xlsx')
# df.to_excel(writer, 'Hoja de datos', index=False)
# writer.save()

ad_fuller_result = adfuller(df['precio_spot'])
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')
# p-value is 2.019150432616161e-15 so we can assume that the trend is stationary

# Seasonal differencing
# df['precio_spot'] = df['precio_spot'].diff(365)
# data = data.drop([1, 2, 3, 4], axis=0).reset_index(drop=True)

plot_pacf(df['precio_spot'])
plot_acf(df['precio_spot'])

# Defining the exogenous variables and the target 
predictors = ['demanda_p48', 'eolica_p48']
target = 'precio_spot'

data_x = pd.get_dummies(df[predictors])
data_y = df.precio_spot

# train = df[df.index < pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
# test = df[df.index > pd.to_datetime("2020-11-01", format='%Y-%m-%d')]

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

# DataFrame conversion
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_val = pd.DataFrame(X_val)
y_val = pd.DataFrame(y_val)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

y_test.info()
y_test.describe()

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(data_y)
axes[0, 0].set_title('Original Series')
plot_acf(data_y, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(data_y.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(data_y.values.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(data_y.values.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(data_y.values.diff().diff().dropna(), ax=axes[2, 1])

plt.show()


p = range(0, 4, 1) # range(start, stop, step)
d = range(0, 3, 1)
q = range(0, 4, 1)
P = range(0, 4, 1)
D = range(0, 2, 1)
Q = range(0, 4, 1)
s = 8760

parameters = product(p, q, P, Q)
parameters_list = list(parameters)
print(len(parameters_list))


result_df = optimize_SARIMA(parameters_list, 1, 1, s, y_train)
result_df


def optimize_SARIMA(parameters_list, d, D, s, exog):
    """
        Return dataframe with parameters, corresponding AIC and SSE
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
        exog - the exogenous variable
    """
    
    results = []
    
    for param in tqdm_notebook(parameters_list):
        try: 
            model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
            
        aic = model.aic
        results.append([param, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)x(P,Q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df


# 2º Option

p = range(0, 3)
d = range(1, 2)
q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 365) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y_train,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}365 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue














best_model = SARIMAX(df['precio_spot'], order=(0, 1, 2), seasonal_order=(0, 1, 2, 4)).fit(dis=-1)
print(best_model.summary())


best_model.plot_diagnostics(figsize=(15,12))





# Now we can start with the predictions
data['arima_model'] = best_model.fittedvalues
data['arima_model'][:4+1] = np.NaN
forecast = best_model.predict(start=data.shape[0], end=data.shape[0] + 8)
forecast = data['arima_model'].append(forecast)
plt.figure(figsize=(15, 7.5))
plt.plot(forecast, color='r', label='model')
plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
plt.plot(data['data'], label='actual')
plt.legend()
plt.show()















# manual

# grid search sarima hyperparameters
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv
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

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter
import statsmodels.api as sm
import matplotlib
import pmdarima as pm

# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend,
    enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]

def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error



# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)


# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
        # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores



# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n','c','t','ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p,d,q), (P,D,Q,m), t]
                                    models.append(cfg)

    return models


if __name__ == '__main__':
    # load dataset
    df = pd.read_csv('dataset.csv',sep = ';', nrows = 8761)
    
    # a = [str(date) for date in df['datetime']]
    # a = [date[:-6] for date in a]
    # date_time = pd.to_datetime(a)
    
    # df.loc[:,['datetime']] = date_time
    # df = df.set_index('datetime') 
    
    df = df.precio_spot
    data = df.values
    # data split
    n_test = 1000
    # model configs
    cfg_list = sarima_configs()
    # grid search
    scores = grid_search(data, cfg_list, n_test)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)

# Best ARIMA model = (2, 1, 4)




