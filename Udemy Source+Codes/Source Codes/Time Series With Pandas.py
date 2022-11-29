"""
Pandas DateTime

"""
"""
Creating Date Time Column

"""
import pandas as pd

pd.date_range('jan 01 2021', periods = 12, freq = 'M')

Date1 = pd.date_range('jan 01 2021', periods = 12, freq = 'M')

Date2 = pd.date_range('jan 01 2021', periods = 4, freq = '3M')

Date3 = pd.date_range('jan 01 2021', periods = 8760, freq = 'H')

"""
Set Date as Index

"""


import numpy as np

Data_set = np.random.randint(1,1000, (8760,2))

Data_Set_DF = pd.DataFrame(Data_set)

Data_Set_DF.set_index(Date3, inplace=True)

df = pd.read_csv('Data_Set.csv')

df = pd.read_csv('Data_Set.csv', index_col= 'DATE', parse_dates= True)

df.info()


df2 = pd.read_csv('Data_Set.csv', index_col= 'DATE')

df2.info()

"""
Resampling

"""

RS_df = df.resample(rule = 'D').mean()


RS_df = df.resample(rule = 'A').mean()
