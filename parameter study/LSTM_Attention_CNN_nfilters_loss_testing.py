#coding=utf-8
import os
import time
import warnings
import numpy as np
import pandas as pd
import operator
from functools import reduce
import h5py
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Convolution1D, MaxPooling1D, Flatten,  Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras_self_attention import SeqSelfAttention
from datetime import datetime
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# filter = 32
np.random.seed(7)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# read the data from excel by pandas
path = r'Daily Gold price.xlsx'
df = pd.read_excel(path, sheet_name=0)

# 利用 pandas 的to_datetime 方法，把 "Date" 列的字符类型数据解析成 datetime对象，然后，把 "Date" 列用作索引。
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
#print(df)


# 制成数据组的函数
def create_dataset(dataset, look_back=1, columns=['Dollar']):
    dataX, dataY = [], []
    for i in range(len(dataset.index)):
        if i < look_back:
            continue
        a = None
        for c in columns:
            b = dataset.loc[dataset.index[i - look_back:i], c].as_matrix()  # 从DataFrame到矩阵的转换
            if a is None:
                a = b  # 如果a是NaN，就把b代替a
            else:
                a = np.append(a, b)  # 否则，b拼到a后面
        dataX.append(a)
        dataY.append(dataset.loc[dataset.index[i - look_back], columns].as_matrix())
    return np.array(dataX), np.array(dataY)


look_back = 7
sc = StandardScaler()  # 标准化数据
df.loc[:, 'Dollar'] = sc.fit_transform(df.Dollar.values.reshape(-1, 1))  # fit.transform()先拟合数据，再标准化
#print(df.loc[:, 'Dollar'])

# Create training data
#train_df = df.loc[df.index < pd.to_datetime('2010-01-01')]
train_df = df.loc[df.index < df.index[int(len(df.index) * 0.8)]]
train_x, train_y = create_dataset(train_df, look_back=look_back)

# Construct the whole LSTM + CNN
model = Sequential()
# LSTM
model.add(LSTM(input_shape=(look_back, 1), input_dim=1, output_dim=6, return_sequences=True))
# model.add(LSTM(input_shape = (look_back,1), input_dim=1, output_dim=6, return_sequences=True))
# model.add(Dense(1))
# model.add(Activation('relu')) # ReLU : y = max(0,x)

# Attention Mechanism
model.add(SeqSelfAttention(attention_activation='sigmoid', name='Attention'))


# CNN
model.add(Convolution1D(input_shape=(look_back, 1),
                        nb_filter=32,
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# model.add(MaxPooling1D(pool_length=2)) # 池化窗口大小

'''model.add(Convolution1D(input_shape = (look_back,1),
                        nb_filter=64,
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))'''
model.add(MaxPooling1D(pool_length=2))

# model.add(Dropout(0.25))

# model.add(Dense(250)) # Dense就是常用的全连接层，250代表output的shape为(*,250)
model.add(Dropout(0.25))
model.add(Activation('relu'))  # ReLU : y = max(0,x)
model.add(Dense(1))
model.add(Activation('linear'))  # Linear : y = x

# training the train data with n epoch
model.compile(loss="mse", optimizer="adam")
result1 = model.fit(np.atleast_3d(np.array(train_x)),
          np.atleast_3d(train_y),
          epochs=1200,
          batch_size=80, verbose=1, shuffle=False)

with open('RSA_filter32_loss.txt','w') as f:
    f.write(str(result1.history))

model.save('RSA_filter32_loss.h5')

# filter = 64

np.random.seed(7)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# read the data from excel by pandas
path = r'Daily Gold price.xlsx'
df = pd.read_excel(path, sheet_name=0)

# 利用 pandas 的to_datetime 方法，把 "Date" 列的字符类型数据解析成 datetime对象，然后，把 "Date" 列用作索引。
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
#print(df)


# 制成数据组的函数
def create_dataset(dataset, look_back=1, columns=['Dollar']):
    dataX, dataY = [], []
    for i in range(len(dataset.index)):
        if i < look_back:
            continue
        a = None
        for c in columns:
            b = dataset.loc[dataset.index[i - look_back:i], c].as_matrix()  # 从DataFrame到矩阵的转换
            if a is None:
                a = b  # 如果a是NaN，就把b代替a
            else:
                a = np.append(a, b)  # 否则，b拼到a后面
        dataX.append(a)
        dataY.append(dataset.loc[dataset.index[i - look_back], columns].as_matrix())
    return np.array(dataX), np.array(dataY)


look_back = 7
sc = StandardScaler()  # 标准化数据
df.loc[:, 'Dollar'] = sc.fit_transform(df.Dollar.values.reshape(-1, 1))  # fit.transform()先拟合数据，再标准化
#print(df.loc[:, 'Dollar'])

# Create training data
#train_df = df.loc[df.index < pd.to_datetime('2010-01-01')]
train_df = df.loc[df.index < df.index[int(len(df.index) * 0.8)]]
train_x, train_y = create_dataset(train_df, look_back=look_back)

# Construct the whole LSTM + CNN
model = Sequential()
# LSTM
model.add(LSTM(input_shape=(look_back, 1), input_dim=1, output_dim=6, return_sequences=True))
# model.add(LSTM(input_shape = (look_back,1), input_dim=1, output_dim=6, return_sequences=True))
# model.add(Dense(1))
# model.add(Activation('relu')) # ReLU : y = max(0,x)

# Attention Mechanism
model.add(SeqSelfAttention(attention_activation='sigmoid', name='Attention'))


# CNN
model.add(Convolution1D(input_shape=(look_back, 1),
                        nb_filter=64,
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# model.add(MaxPooling1D(pool_length=2)) # 池化窗口大小

'''model.add(Convolution1D(input_shape = (look_back,1),
                        nb_filter=64,
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))'''
model.add(MaxPooling1D(pool_length=2))

# model.add(Dropout(0.25))

# model.add(Dense(250)) # Dense就是常用的全连接层，250代表output的shape为(*,250)
model.add(Dropout(0.25))
model.add(Activation('relu'))  # ReLU : y = max(0,x)
model.add(Dense(1))
model.add(Activation('linear'))  # Linear : y = x

# training the train data with n epoch
model.compile(loss="mse", optimizer="adam")
result2 =model.fit(np.atleast_3d(np.array(train_x)),
          np.atleast_3d(train_y),
          epochs=1200,
          batch_size=80, verbose=1, shuffle=False)

with open('RSA_filter64_loss.txt','w') as f:
    f.write(str(result2.history))

model.save('RSA_filter64_loss.h5')


# filter = 128

np.random.seed(7)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# read the data from excel by pandas
path = r'Daily Gold price.xlsx'
df = pd.read_excel(path, sheet_name=0)

# 利用 pandas 的to_datetime 方法，把 "Date" 列的字符类型数据解析成 datetime对象，然后，把 "Date" 列用作索引。
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
#print(df)


# 制成数据组的函数
def create_dataset(dataset, look_back=1, columns=['Dollar']):
    dataX, dataY = [], []
    for i in range(len(dataset.index)):
        if i < look_back:
            continue
        a = None
        for c in columns:
            b = dataset.loc[dataset.index[i - look_back:i], c].as_matrix()  # 从DataFrame到矩阵的转换
            if a is None:
                a = b  # 如果a是NaN，就把b代替a
            else:
                a = np.append(a, b)  # 否则，b拼到a后面
        dataX.append(a)
        dataY.append(dataset.loc[dataset.index[i - look_back], columns].as_matrix())
    return np.array(dataX), np.array(dataY)


look_back = 7
sc = StandardScaler()  # 标准化数据
df.loc[:, 'Dollar'] = sc.fit_transform(df.Dollar.values.reshape(-1, 1))  # fit.transform()先拟合数据，再标准化
#print(df.loc[:, 'Dollar'])

# Create training data
#train_df = df.loc[df.index < pd.to_datetime('2010-01-01')]
train_df = df.loc[df.index < df.index[int(len(df.index) * 0.8)]]
train_x, train_y = create_dataset(train_df, look_back=look_back)

# Construct the whole LSTM + CNN
model = Sequential()
# LSTM
model.add(LSTM(input_shape=(look_back, 1), input_dim=1, output_dim=6, return_sequences=True))
# model.add(LSTM(input_shape = (look_back,1), input_dim=1, output_dim=6, return_sequences=True))
# model.add(Dense(1))
# model.add(Activation('relu')) # ReLU : y = max(0,x)

# Attention Mechanism
model.add(SeqSelfAttention(attention_activation='sigmoid', name='Attention'))

# CNN
model.add(Convolution1D(input_shape=(look_back, 1),
                        nb_filter=128,
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# model.add(MaxPooling1D(pool_length=2)) # 池化窗口大小

'''model.add(Convolution1D(input_shape = (look_back,1),
                        nb_filter=64,
                        filter_length=2,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))'''
model.add(MaxPooling1D(pool_length=2))

# model.add(Dropout(0.25))

# model.add(Dense(250)) # Dense就是常用的全连接层，250代表output的shape为(*,250)
model.add(Dropout(0.25))
model.add(Activation('relu'))  # ReLU : y = max(0,x)
model.add(Dense(1))
model.add(Activation('linear'))  # Linear : y = x

# training the train data with n epoch
model.compile(loss="mse", optimizer="adam")
result3 = model.fit(np.atleast_3d(np.array(train_x)),
          np.atleast_3d(train_y),
          epochs=1200,
          batch_size=80, verbose=1, shuffle=False)

with open('RSA_filter128_loss.txt','w') as f:
    f.write(str(result3.history))

model.save('RSA_filter128_loss.h5')



plt.grid(ls='--')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.plot(result1.epoch, result1.history['loss'], 'red',lw=2, label = '32 CNN filters')
plt.plot(result2.epoch, result2.history['loss'], 'green', lw=2, label = '64 CNN filters')
plt.plot(result3.epoch, result3.history['loss'], 'blue', lw=2, label = '128 CNN filters')
plt.legend()
plt.savefig("RSA_nfilters_loss_result.eps", format='eps', dpi=1000)
plt.show()