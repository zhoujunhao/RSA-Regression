import os
import warnings
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from datetime import datetime
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

np.random.seed(7)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# read the data from excel by pandas
#path = r'D:\Final Year Project\Daily Gold price.xlsx'
path = r'palladium-prices-historical-chart-data_gold.xlsx'

df = pd.read_excel(path, sheet_name=0)

# 利用 pandas 的to_datetime 方法，把 "Date" 列的字符类型数据解析成 datetime对象，然后，把 "Date" 列用作索引。
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
print(df)

# Set up SVR model
for c in df.columns:
    df[c+'_ret'] = df[c].pct_change().fillna(0)
svr_poly = SVR(kernel='poly', C=100, degree=2, verbose=1)

def create_dataset(dataset, look_back=1, columns = ['Dollar']):
    dataX, dataY = [], []
    for i in range(len(dataset.index)):
        if i <= look_back:
            continue
        a = None
        for c in columns:
            b = dataset.loc[dataset.index[i-look_back:i], c].as_matrix()
            if a is None:
                a = b
            else:
                a = np.append(a,b)
        dataX.append(a)
        dataY.append(dataset.loc[dataset.index[i], columns].as_matrix())
    return np.array(dataX), np.array(dataY)

look_back = 7
sc = StandardScaler() # 标准化数据
df.loc[:, 'Dollar'] = sc.fit_transform(df.Dollar.values.reshape(-1,1)) # fit.transform()先拟合数据，再标准化

#train_df = df.loc[df.index < pd.to_datetime('2012-01-01')]
#val_df = train_df.loc[train_df.index >= pd.to_datetime('2011-01-01')]
#train_df = train_df.loc[train_df.index < pd.to_datetime('2012-01-01')] # if train data < 2011-/-/ the prediction line will inverse, why?
train_df = df.loc[df.index < df.index[int(len(df.index)*0.8)]] #用80%的数据作为训练数据


predictors = ['Dollar']#, 'InterestRate']
train_x, train_y = create_dataset(train_df, look_back=look_back, columns=predictors)
#val_x, val_y = create_dataset(val_df, look_back=look_back, columns=predictors)

svr_poly = svr_poly.fit(train_x, train_y)

# Make prediction
df['Pred'] = df.loc[df.index[0], 'Dollar']
for i in range(len(df.index)):
    if i < look_back:
        continue
    a = None
    for c in predictors:
        b = df.loc[df.index[i-look_back:i], c].as_matrix()
        if a is None:
            a = b
        else:
            a = np.append(a,b)
        a = a
    y = svr_poly.predict(a.reshape(1,look_back*len(predictors))) # 制作测试数据并把其制成矩阵，然后将其放入已经训练完的模型中。此处用的测试数据是所有数据
    df.loc[df.index[i], 'Pred']=y[0]

df.loc[:, 'Dollar'] = sc.inverse_transform(df.loc[:, 'Dollar']) #将标准化后是数据转换为原始数据
df.loc[:, 'Pred'] = sc.inverse_transform(df.loc[:, 'Pred'])

def mape(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape

# present the line chart and some parameters like MSE, which reflects the accuracy of the model in sample or out sample
plt.plot(df.Dollar, 'b', label='Price')
#plt.plot(df.loc[df.index < pd.to_datetime('2012-01-01'), 'Pred'], 'r', label='Insample Prediction')
#plt.plot(df.loc[df.index >= pd.to_datetime('2012-01-01'), 'Pred'], 'g', label='Outsample Prediction')
#plt.plot(df.loc[df.index < df.index[int(len(df.index)*0.8)], 'Pred'], 'r', label='Insample Prediction')
#plt.plot(df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Pred'], 'g', label='Insample Prediction')
print('The RMSE is ','%e'%sqrt(mean_squared_error(df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Dollar'], df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Pred'])))
print('The RMAE is ','%e'%sqrt(mean_absolute_error(df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Dollar'], df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Pred'])))
print('The MAPE is ','%e'%mape(df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Dollar'], df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Pred']))
#print('%e'%mean_squared_error(df.loc[df.index < pd.to_datetime('2012-01-01'),'Dollar'],df.loc[df.index < pd.to_datetime('2012-01-01'),'Pred']))
#print('%e'%mean_squared_error(df.loc[df.index >= pd.to_datetime('2012-01-01'),'Dollar'],df.loc[df.index >= pd.to_datetime('2012-01-01'),'Pred']))
#plt.legend()
#plt.show()