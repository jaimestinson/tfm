#######  Time series forecasting with Tensorflow - MLP

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



df = pd.read_csv('dataset.csv',sep = ';', nrows = 17545)

a = [str(date) for date in df['datetime']]
a = [date[:-6] for date in a]
date_time = pd.to_datetime(a)

df.loc[:,['datetime']] = date_time
df = df.set_index('datetime') 

timestamp_s = date_time.map(pd.Timestamp.timestamp)
day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

plt.plot(np.array(df['Day sin'])[:25])
plt.plot(np.array(df['Day cos'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')

df.index
df.index.freq = 'H'

# Evolution of the features over time:
plot_cols = ['demanda_p48', 'eolica_p48', 'precio_spot']
plot_features = df[plot_cols]
_ = plot_features.plot(subplots=True)

# Correlation between the exogenous variables
sns.heatmap(df.corr())

# ACF and PACF plots
plot_acf(df.precio_spot, lags = 50)
plot_pacf(df.precio_spot, lags = 50)

# Defining the exogenous variables and the target 
predictors = ['demanda_p48', 'eolica_p48', 'Day sin', 'Day cos', 'Year sin', 'Year cos']
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





# %%Baseline models
result_df = pd.DataFrame(index=X_test.index, columns=['Random','Naive_Mean','LR','Actual'])
result_df.Actual = y_test

# Method: Naive Mean
result_df.Naive_Mean = y_train.mean()

# Method: Random
result_df.Random = np.random.uniform(y_train.min(), y_train.max(),y_test.shape)

#Method: LR
df_lm = LinearRegression()
df_lm.fit(X_train, y_train)
result_df.LR =df_lm.predict(X_test)
# result_df is an array with the predictions for the test set
# result_df = pred_test

result_df['|Random-Actual|'] = abs(result_df.Random-result_df.Actual)
result_df['|Naive_Mean-Actual|'] = abs(result_df.Naive_Mean-result_df.Actual)
result_df['|LR-Actual|'] = abs(result_df.LR-result_df.Actual)


table = result_df[['|Random-Actual|','|Naive_Mean-Actual|','|LR-Actual|']]

# plt.figure(figsize=(2,10))
# sns.heatmap(table, center=table.mean().mean(),cmap="Greys")
# plt.show()


metric_df = pd.DataFrame(index = ['ME','RMSE', 'MAE','MAPE'] ,
                         columns = ['Random','Naive_Mean','LR'])

n_test = len(result_df)

for m in metric_df.columns:
    metric_df.at['ME',m]= np.sum((result_df.Actual - result_df[m]))/n_test
    metric_df.at['RMSE',m]= np.sqrt(np.sum(result_df.apply(lambda r: (r.Actual - r[m])**2,axis=1))/n_test)
    metric_df.at['MAE',m] = np.sum(abs(result_df.Actual - result_df[m]))/n_test
    metric_df.at['MAPE',m] = np.sum(result_df.apply(lambda r:abs(r.Actual-r[m])/r.Actual,axis=1))/n_test*100
metric_df


#### MLP 

df_mlp = MLPRegressor(hidden_layer_sizes = 5, max_iter = 2000)
df_mlp.fit(X_train, y_train)

result_df['MLP'] = df_mlp.predict(X_test)
result_df['|MLP-Actual|'] = abs(result_df.MLP-result_df.Actual)

table = result_df[['|Random-Actual|','|Naive_Mean-Actual|','|LR-Actual|','|MLP-Actual|']]

# plt.figure(figsize=(2,10))
# sns.heatmap(table, center=table.mean().mean(),cmap="Greys")
# plt.show()




m='MLP'

metric_df.at['ME',m]= np.sum((result_df.Actual - result_df[m]))/n_test
metric_df.at['RMSE',m]= np.sqrt(np.sum(result_df.apply(lambda r: (r.Actual - r[m])**2,axis=1))/n_test)
metric_df.at['MAE',m] = np.sum(abs(result_df.Actual - result_df[m]))/n_test
metric_df.at['MAPE',m] = np.sum(result_df.apply(lambda r:abs(r.Actual-r[m])/r.Actual,axis=1))/n_test*100

metric_df

# Now we tune the MLP

# 1 
# Create tuning (validation) set: devide the trainset

X_train_s, X_tune, y_train_s, y_tune = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1)

print('X_train Shape: ', X_train.shape)
print('y_train Shape: ', y_train.shape)

print('X_train_s Shape: ', X_train_s.shape)
print('X_tune Shape: ', X_tune.shape)
print('y_train_s Shape: ', y_train_s.shape)
print('y_tune Shape: ', y_tune.shape)


# Tune for - activation and solver

num_repetition = 5
# Create a placeholder for experimentations
activation_options = ['identity', 'logistic', 'tanh', 'relu']
solver_options = ['lbfgs','sgd','adam']

my_index = pd.MultiIndex.from_product([activation_options,solver_options],
                                     names=('activation', 'solver'))

tune_df = pd.DataFrame(index = my_index,
                       columns=['R{}'.format(i) for i in range(num_repetition)])

tune_df

n = len(y_tune)
for activation_o in activation_options:
    for solver_o in solver_options:
        for rep in tune_df.columns:
            df_mlp = MLPRegressor(hidden_layer_sizes=(10), max_iter=5000,
                                   activation=activation_o,solver=solver_o)

            df_mlp.fit(X_train_s, y_train_s)
            y_tune_predict = df_mlp.predict(X_tune)
            RSME = np.sqrt(np.sum((y_tune_predict - y_tune)**2)/n)
            
            tune_df.at[(activation_o,solver_o),rep] = RSME

# After a lot warnings, we get to the most accurate activation and solver:
# relu and adam

tune_df['Mean'] = tune_df[
    ['R{}'.format(i) for i in range(num_repetition)]].mean(axis=1)
tune_df['Min'] = tune_df[
    ['R{}'.format(i) for i in range(num_repetition)]].min(axis=1)
tune_df.sort_values('Min')


# Best activation and solver: relu and adam

# 2
# Tune for hidden_layer_sizes

# This code is basically creating a list of 15 one layered ANN ([1] - [15]) 
# and 100 two layered ANNs ([1,1] - [10,10])

PossibleNetStrct = []
PossibleNetString = []

for i in range(1,16):
    netStruct = [i]
    PossibleNetStrct.append(netStruct)
    PossibleNetString.append(str(netStruct))
    print('Network Structure:', netStruct)
    
for i in range(1,11):
    for j in range(1,11):
        netStruct = [i,j]
        PossibleNetStrct.append(netStruct)
        PossibleNetString.append(str(netStruct))
        print('Network Structure:', netStruct)


tune_df = pd.DataFrame(np.nan, index =PossibleNetString,
                      columns = ['R{}'.format(i) for i in range(num_repetition)])

n = len(y_tune)

for i,netStr in enumerate(PossibleNetStrct):
    RowName = PossibleNetString[i]
    for rep in tune_df.columns:
        
        df_mlp = MLPRegressor(hidden_layer_sizes=netStr, activation = 'relu', 
                       solver='adam', max_iter=5000)
        df_mlp.fit(X_train_s, y_train_s)
        
        y_tune_predict = df_mlp.predict(X_tune)
        RSME = np.sqrt(np.sum((y_tune_predict - y_tune)**2)/n)
        
        tune_df.at[RowName, rep] = RSME
        
    print(netStr)

tune_df['Mean'] = tune_df[
    ['R{}'.format(i) for i in range(num_repetition)]].mean(axis=1)
tune_df['Min'] = tune_df[
    ['R{}'.format(i) for i in range(num_repetition)]].min(axis=1)
tune_df.sort_values('Min')

tune_df.sort_values('Mean')

# (4) neural network
# one hidden layer with 4 neurons





# 3
# Tune for learning_rate, learning_rate_init

LR_options = ['constant','invscaling','adaptive']
LRI_options = [0.0001,0.001,0.005,0.01,0.05,0.1]

my_index = pd.MultiIndex.from_product([LR_options,LRI_options],
                                     names=('LR', 'rate'))

tune_df = pd.DataFrame(index = my_index,
                       columns=['R{}'.format(i) for i in range(num_repetition)])

tune_df

for LR_o in LR_options:
    for LRI_o in LRI_options:
        for rep in tune_df.columns:
            df_mlp = MLPRegressor(hidden_layer_sizes=(4), max_iter=2000,
                                   activation='relu',solver='adam',
                                   learning_rate=LR_o, learning_rate_init= LRI_o)

            df_mlp.fit(X_train_s, y_train_s)
            y_tune_predict = df_mlp.predict(X_tune)
            RSME = np.sqrt(np.sum((y_tune_predict - y_tune)**2)/n)
            
            tune_df.at[(LR_o,LRI_o),rep] = RSME
        print((LR_o,LRI_o))



tune_df['Mean'] = tune_df[
    ['R{}'.format(i) for i in range(num_repetition)]].mean(axis=1)
tune_df['Min'] = tune_df[
    ['R{}'.format(i) for i in range(num_repetition)]].min(axis=1)
tune_df.sort_values('Mean')

# LR adaptive 0.0001





# 4
# Tune for max_iter, shuffle
max_iterations_options = [500,1000,2000,5000,10000]
shuffle_options = [True,False]

my_index = pd.MultiIndex.from_product([max_iterations_options,shuffle_options],
                                     names=('Max Iterations', 'shuffle'))

tune_df = pd.DataFrame(index = my_index,
                       columns=['R{}'.format(i) for i in range(num_repetition)])

tune_df


for max_iterations_o in max_iterations_options:
    for shuffle_o in shuffle_options:
        for rep in tune_df.columns:
            df_mlp = MLPRegressor(hidden_layer_sizes=(7), max_iter=max_iterations_o,
                                   activation='relu',solver='adam',
                                   learning_rate='adaptive', learning_rate_init= 0.0001,
                                   shuffle = shuffle_o)

            df_mlp.fit(X_train_s, y_train_s)
            y_tune_predict = df_mlp.predict(X_tune)
            RSME = np.sqrt(np.sum((y_tune_predict - y_tune)**2)/n)
            
            tune_df.at[(max_iterations_o,shuffle_o),rep] = RSME
        print((max_iterations_o,shuffle_o))


tune_df['Mean'] = tune_df[
    ['R{}'.format(i) for i in range(num_repetition)]].mean(axis=1)
tune_df['Min'] = tune_df[
    ['R{}'.format(i) for i in range(num_repetition)]].min(axis=1)
tune_df.sort_values('Mean')


# max iterations: 2000 Shuffle: False




# 5
# Tune for alpha
alpha_options =[0.00001,0.00005,0.0001,0.0005,0.001,0.005]

tune_df = pd.DataFrame(index = alpha_options,
                       columns=['R{}'.format(i) for i in range(num_repetition)])

tune_df



for alpha_o in alpha_options:
    for rep in tune_df.columns:
        df_mlp = MLPRegressor(hidden_layer_sizes=(7), max_iter=2000,
                               activation='relu',solver='adam', learning_rate='adaptive',
                               learning_rate_init= 0.0001, shuffle = False, alpha=alpha_o)
        df_mlp.fit(X_train_s, y_train_s)
        y_tune_predict = df_mlp.predict(X_tune)
        RSME = np.sqrt(np.sum((y_tune_predict - y_tune)**2)/n)
        tune_df.at[alpha_o,rep] = RSME
    print(alpha_o)


tune_df['Mean'] = tune_df[
    ['R{}'.format(i) for i in range(num_repetition)]].mean(axis=1)
tune_df['Min'] = tune_df[
    ['R{}'.format(i) for i in range(num_repetition)]].min(axis=1)
tune_df.sort_values('Min')


# alpha = 0.0005



# 6
# Tune for Randomness
random_options = range(1,10)

tune_df = pd.DataFrame(index = random_options,
                       columns=['RSME'])
tune_df


for random_o in random_options:
    df_mlp = MLPRegressor(hidden_layer_sizes=(7), max_iter=2000,
                               activation='relu',solver='adam', learning_rate='adaptive',
                               learning_rate_init= 0.0001, shuffle = False, alpha = 0.0005, random_state=random_o)
    df_mlp.fit(X_train_s, y_train_s)
    y_tune_predict = df_mlp.predict(X_tune)
    RSME = np.sqrt(np.sum((y_tune_predict - y_tune)**2)/n)
    tune_df.at[random_o,'RSME'] = RSME
    print(random_o)


tune_df

# Random_state = 1






# %%Train the tuned MLP on train set

# 1.Create tuning (validation) set: devide the trainset

# X_train_s, X_tune, y_train_s, y_tune = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1)

# # DataFrame conversion
# X_train_s = pd.DataFrame(X_train_s)
# X_tune = pd.DataFrame(X_tune)
# y_train_s = pd.DataFrame(y_train_s)
# y_tune = pd.DataFrame(y_tune)

# Training set
print('X_train Shape: ', X_train.shape)
print('y_train Shape: ', y_train.shape)

# Validation set
print('X_val Shape: ', X_val.shape)
print('y_val Shape: ', y_val.shape)

# Test set
print('x_test Shape: ', X_test.shape)
print('y_test Shape: ', y_test.shape)

df_mlp = MLPRegressor(hidden_layer_sizes=(7), max_iter=2000,
                               activation='relu',solver='adam', learning_rate='adaptive',
                               learning_rate_init= 0.0001, shuffle = False, alpha = 0.0005, random_state=1)

df_mlp.fit(X_train, y_train)

predictions_val = df_mlp.predict(X_val)
predictions_val = pd.DataFrame(predictions_val)
predictions_test = df_mlp.predict(X_test)
predictions_test = pd.DataFrame(predictions_test)


y_val['Prediction'] = predictions_val.values
y_val['Prediction'] = predictions_val.to_numpy()

y_test['Prediction'] = predictions_test.values
y_test['Prediction'] = predictions_test.to_numpy()

import math 
from sklearn.metrics import mean_squared_error

RMSE_val = math.sqrt(mean_squared_error(y_val.precio_spot, y_val.Prediction))

RMSE_test = math.sqrt(mean_squared_error(y_test.precio_spot, y_test.Prediction))

RMSE_val
RMSE_test

def mean_absolute_percentage_error (y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))

MAPE_val = mean_absolute_percentage_error(y_val.precio_spot, y_val.Prediction)
MAPE_test = mean_absolute_percentage_error(y_test.precio_spot, y_test.Prediction)
MAPE_val
MAPE_test


# df_mlp.score(X_val, y_val.precio_spot)

# test_accuracy = accuracy_score(y_test.Prediction, y_test.precio_spot)


# y_test.shape
# y_test.sort_index(axis = 0)
# y_test.sort_index()


y_train.info()
y_test.info()



fig, ax = plt.subplots(figsize=(9, 4))
plt.xlabel('Date')
plt.ylabel('EP')
plt.title('EPF')
y_train.precio_spot.plot(ax=ax,  color = "black", label='Training')
y_val.precio_spot.plot(ax=ax,  color = "blue", label='Validation')
y_val.Prediction.plot(ax=ax,  color = "green", label='Predictions_val')
y_test.precio_spot.plot(ax=ax, color = "red", label='Test')
y_test.Prediction.plot(ax=ax, color='green', label='Predictions_test')
ax.legend()

# dudas

# autoregresivo? 
# vectorizado o single-output?


m='MLP_tuned'

metric_df.at['ME',m]= np.sum((result_df.Actual - result_df[m]))/n_test
metric_df.at['RMSE',m]= np.sqrt(np.sum(result_df.apply(lambda r: (r.Actual - r[m])**2,axis=1))/n_test)
metric_df.at['MAE',m] = np.sum(abs(result_df.Actual - result_df[m]))/n_test
metric_df.at['MAPE',m] = np.sum(result_df.apply(lambda r:abs(r.Actual-r[m])/r.Actual,axis=1))/n_test*100

metric_df

from sklearn.metrics import accuracy_score

predictions_train = df_mlp.predict(X_train)
predictions_test = result_df['MLP_tuned']
train_score = accuracy_score(predictions_train, train_labels)

print("score on train data: ", train_score)
test_score = accuracy_score(predictions_test, test_labels)
print("score on test data: ", test_score)



print("Weights between input and first hidden layer:")
print(df_mlp.coefs_[0])
print("\nWeights between first hidden and second hidden layer:")
print(df_mlp.coefs_[1])