import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,LocallyConnected1D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Sequential
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,GaussianNoise,AveragePooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D
import numpy as np
import scipy.io as sio
from keras import backend as K

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
mat=sio.loadmat('train_feature.mat',squeeze_me=True)
train_feature=mat['train_feature']

mat=sio.loadmat('train_label.mat',squeeze_me=True)
train_label=mat['train_label']
train_label=keras.utils.to_categorical(train_label,num_classes=10)

mat=sio.loadmat('test_feature.mat',squeeze_me=True)
test_feature=mat['test_feature']

mat=sio.loadmat('test_label.mat',squeeze_me=True)
test_label=mat['test_label']
test_label=keras.utils.to_categorical(test_label,num_classes=10)
print('##########2')
train_features=np.reshape(train_feature,(50000,32,32,3))
test_features=np.reshape(test_feature,(10000,32,32,3))
train_features=train_features/255
test_features=test_features/255
print('##########3')

model=Sequential()
model.add(Conv2D(16,kernel_size,padding='valid',activation='relu',input_shape=input_shape))
model.add(Conv2D(32,kernel_size,activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Conv2D(64,kernel_size,activation='selu'))
model.add(AveragePooling2D(pool_size=pool_size))
model.add(Conv2D(128,kernel_size,activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Flatten())
model.add(Dense(units=300,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=200,activation='selu'))
model.add(Dropout(0.1))
model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.1))
model.add(GaussianNoise(0.01))
model.add(Dense(units=10,activation='softmax'))

keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

model.compile(loss='binary_crossentropy',optimizer='adamax',metrics=['accuracy'])
model.fit(train_features,train_label,epochs=50,batch_size=1000)

score=model.evaluate(test_features,test_label)
print(score)
