# FACEBOOK PROPHET

import pandas as pd
from prophet import Prophet

df = pd.read_csv('dataset.csv',sep = ';', nrows = 17544)

a = [str(date) for date in df['datetime']]
a = [date[:-6] for date in a]
date_time = pd.to_datetime(a)

df.loc[:,['datetime']] = date_time
df = df.set_index('datetime') 
df = df.precio_spot

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
future.tail()


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)

plot_components_plotly(m, forecast)
