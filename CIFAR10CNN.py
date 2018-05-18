
from __future__ import print_function
import os

import datetime
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,GaussianNoise, Lambda
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import scipy.io as sio
from keras.datasets import cifar10
#from PoolFcn import TracePool as TP
import tensorflow as tf
K.set_learning_phase(1) 


now = datetime.datetime.now

img_rows,img_cols=32,32
num_classes=10
pool_size=2
filters=32
kernel_size=3
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)
print('##########1')

(train_feature, train_label), (test_feature, test_label) = cifar10.load_data()

train_label=keras.utils.to_categorical(train_label,num_classes)
test_label=keras.utils.to_categorical(test_label,num_classes)
print('##########2')
train_features=np.reshape(train_feature,(50000,32,32,3))
test_features=np.reshape(test_feature,(10000,32,32,3))
train_features=train_features/255
test_features=test_features/255

print('Define model')
model=Sequential()
print('1')
model.add(Conv2D(16,kernel_size,padding='valid',activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size,strides=(1,1)))
model.add(Conv2D(32,kernel_size,padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size,strides=(1,1)))
model.add(Conv2D(64,kernel_size,padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size,strides=(1,1)))

print('3')
model.add(Flatten())
print('4')

model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(units=10,activation='softmax'))
print('5')
print('Start Compiling')
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
model.compile(loss='kullback_leibler_divergence',optimizer='adadelta',metrics=['accuracy'])
print('Done Compiling')
model.fit(train_features,train_label,epochs=20,batch_size=500)
print('Done Fitting')
score=model.evaluate(test_features,test_label,batch_size=100)
print('Done evaluation')
print(score)

