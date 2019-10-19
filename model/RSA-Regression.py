# -*- coding: UTF-8 -*-
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import warnings
import numpy as np
import pandas as pd
import operator
from functools import reduce
import h5py
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Convolution1D, MaxPooling1D, Flatten,  Embedding,Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D, merge
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

np.random.seed(7)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# read the data from excel by pandas
#path = r'Daily Gold price.xlsx'
path = r'palladium-prices-historical-chart-data_gold.xlsx'
df = pd.read_excel(path, sheet_name=0)

# 利用 pandas 的to_datetime 方法，把 "Date" 列的字符类型数据解析成 datetime对象，然后，把 "Date" 列用作索引。
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
print(df)

# 制成数据组的函数
def create_dataset(dataset, look_back=1, columns = ['Dollar']):
    dataX, dataY = [], []
    for i in range(len(dataset.index)):
        if i < look_back:
            continue
        a = None
        for c in columns:
            b = dataset.loc[dataset.index[i-look_back:i], c].as_matrix() # 从DataFrame到矩阵的转换
            if a is None:
                a = b # 如果a是NaN，就把b代替a
            else:
                a = np.append(a,b) #否则，b拼到a后面
        dataX.append(a)
        dataY.append(dataset.loc[dataset.index[i-look_back], columns].as_matrix())
    return np.array(dataX), np.array(dataY)

look_back = 7 # 10, 13
sc = StandardScaler() # 标准化数据
df.loc[:, 'Dollar'] = sc.fit_transform(df.Dollar.values.reshape(-1,1)) # fit.transform()先拟合数据，再标准化
print(df.loc[:, 'Dollar'])

# Create training data
#train_df = df.loc[df.index < pd.to_datetime('2010-01-01')]
train_df = df.loc[df.index < df.index[int(len(df.index)*0.8)]]
train_x, train_y = create_dataset(train_df, look_back=look_back)

# Construct the whole LSTM + CNN
model = Sequential()
# LSTM
model.add(LSTM(input_shape = (look_back, 1), input_dim=1, output_dim=6,  return_sequences=True))

#model.add(LSTM(input_shape = (look_back,1), input_dim=1, output_dim=6, return_sequences=True))
#model.add(Dense(1))
#model.add(Activation('relu')) # ReLU : y = max(0,x)

# Attention Mechanism
model.add(SeqSelfAttention(kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,attention_activation='sigmoid', name='Attention'))
#model.add(SeqSelfAttention(attention_activation='sigmoid', name='Attention'))

# CNN
model.add(Convolution1D(input_shape = (look_back,1),
                        nb_filter=64,# 32,128
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
#model.add(MaxPooling1D(pool_length=2)) # 池化窗口大小

'''model.add(Convolution1D(input_shape = (look_back,1),
                        nb_filter=64,
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))'''
model.add(MaxPooling1D(pool_length=2))

#model.add(Dropout(0.25))

#model.add(Dense(250)) # Dense就是常用的全连接层，250代表output的shape为(*,250)
model.add(Dropout(0.25))
model.add(Activation('relu')) # ReLU : y = max(0,x)
model.add(Dense(1))
model.add(Activation('linear')) # Linear : y = x

# Print whole structure of the model
print(model.summary())

# training the train data with n epoch
model.compile(loss="mse", optimizer="adam") # adam, rmsprop
result = model.fit(np.atleast_3d(np.array(train_x)),
          np.atleast_3d(train_y),
          epochs=100,
          batch_size=80, verbose=1, shuffle=False)


with open('rsa_Palladium.txt','w') as f:
    f.write(str(result.history))

model.save('rsa_Palladium.h5')


# Make prediction and specify on the line chart
predictors = ['Dollar']
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
    y = model.predict(a.reshape(1,look_back*len(predictors),1)) # 制作测试数据并把其制成矩阵，然后将其放入已经训练完的模型中。此处用的测试数据是所有数据
    df.loc[df.index[i], 'Pred']=y[0][0]

df.loc[:, 'Dollar'] = sc.inverse_transform(df.loc[:, 'Dollar']) #将标准化后是数据转换为原始数据
df.loc[:, 'Pred'] = sc.inverse_transform(df.loc[:, 'Pred'])

def mape(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape

print('The RMSE is ','%e'%sqrt(mean_squared_error(df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Dollar'], df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Pred'])))
print('The RMAE is ','%e'%sqrt(mean_absolute_error(df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Dollar'], df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Pred'])))
print('The MAPE is ','%e'%mape(df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Dollar'], df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Pred']))

# present the line chart and some parameters like MSE, which reflects the accuracy of the model in sample or out sample
plt.grid(ls='--')
plt.plot(df.loc[df.index < df.index[int(len(df.index)*0.8)], 'Pred'], 'orange', label = 'Insample Prediction')
plt.plot(df.loc[df.index >= df.index[int(len(df.index)*0.8)], 'Pred'], 'g', label = 'Outsample Prediction')
plt.plot(df.Dollar ,'b', label = 'Price')
plt.xlabel('Date')
plt.ylabel('USD/oz')
#print('%e'%mean_squared_error(df.loc[df.index < pd.to_datetime('2010-01-01'),'Dollar'],df.loc[df.index < pd.to_datetime('2010-01-01'),'Pred']))
#print('%e'%mean_squared_error(df.loc[df.index >= pd.to_datetime('2010-01-01'),'Dollar'],df.loc[df.index >= pd.to_datetime('2010-01-01'),'Pred']))


plt.legend(loc='2')
plt.savefig("rsa_Palladium.eps", format='eps', dpi=1000)
plt.show()


'''
# sketch loss
#plt.cla() # clear the axis
plt.grid(ls='--')
plt.plot(result.epoch,result.history['loss'],label='LOSS',c='r',lw=3)
#plt.scatter(result.epoch,result.history['loss'],s=15,c='r')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right', frameon=True, edgecolor='black')
plt.savefig("LC_loss.eps", format='eps', dpi=1000)
plt. close(0)
'''
