import os
import warnings
import numpy as np
import pandas as pd
#from keras.layers.core import Dense, Activation, Dropout
#from keras.layers.recurrent import LSTM
#from keras.models import Sequential
from datetime import datetime
import matplotlib.pyplot as plt
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

np.random.seed(7)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# read the data from excel by pandas
path = r'Daily Gold price.xlsx'
#path = r'palladium-prices-historical-chart-data_gold.xlsx'

d_f = pd.read_excel(path, sheet_name=0)


d_f['Date'] = pd.to_datetime(d_f['Date'])
d_f.set_index('Date', inplace=True)
print(d_f)

for c in d_f.columns:
    d_f[c + '_ret'] = d_f[c].pct_change(2).fillna(0)

sc = StandardScaler()
# 402
# 113
# 302
step = 500
for ai in range(0, len(d_f), step):
    df = d_f.loc[d_f.index[ai:ai + step], :]
    x = df['Dollar_ret']
    x_min = min(x)
    x_max = max(x)
    #    x = (x - x_min)/(x_max-x_min)
    #    sc = StandardScaler()
    #    x = sc.fit_transform(x)
    ##based on model AIC
    min_aic = np.inf
    best_params = (0, 0, 0)
    for i in range(5):
        for j in range(5):
            for k in range(5):
                try:
                    arima = ARIMA(x, order=(i, j, k)).fit()
                    #                    print (i,j,k, arima.aic)
                    if arima.aic < min_aic:
                        min_aic = arima.aic
                        best_params = (i, j, k)
                except:
                    pass
    #
    arima = ARIMA(x, order=best_params).fit()
    print('AIC of ARIMA model', arima.aic)
    print('Params of ARIMA model', best_params)

    x_pred = arima.fittedvalues
    #    x_pred = x_pred*(x_max - x_min) + x_min
    d_f.loc[df.index, 'Pred_ret'] = x_pred  # arima.fittedvalues
df = d_f
df['Pred'] = df.Dollar
for j in range(len(df.index)):
    if j < 2:
        continue
    i = df.index[j]
    prev = df.index[j - 2]
    df.loc[i, 'Pred'] = df.loc[prev, 'Pred'] * (1 + df.loc[i, 'Pred_ret'])

def mape(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape

# present the line chart and some parameters like MSE, which reflects the accuracy of the model in sample or out sample
#plt.plot(df.loc[df.index < pd.to_datetime('2010-01-01'),'Pred'], 'r', label = 'Insample Prediction')
#plt.plot(df.loc[df.index >= pd.to_datetime('2010-01-01'),'Pred'], 'g', label = 'Outsample Prediction')
#plt.plot(df.Dollar ,'b', label = 'Price')
#print('%e'%mean_squared_error(df.loc[df.index < pd.to_datetime('2010-01-01'),'Dollar'],df.loc[df.index < pd.to_datetime('2010-01-01'),'Pred']))
#print('%e'%mean_squared_error(df.loc[df.index >= pd.to_datetime('2010-01-01'),'Dollar'],df.loc[df.index >= pd.to_datetime('2010-01-01'),'Pred']))
print('The RMSE is ','%e'%sqrt(mean_squared_error(df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Dollar'], df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Pred'])))
print('The RMAE is ','%e'%sqrt(mean_absolute_error(df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Dollar'], df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Pred'])))
print('The MAPE is ','%e'%mape(df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Dollar'], df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Pred']))

#plt.legend()
#plt.show()
