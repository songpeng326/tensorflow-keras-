#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:02:00 2018

@author: lab326
"""
from __future__ import print_function

import os
import numpy as np
from keras.layers import merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D,AveragePooling2D
from keras.layers.core import Dense, Activation,Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
from sklearn.cross_validation import train_test_split 
import keras.backend as K
import tensorflow as tf
from keras.utils import np_utils
import h5py
from PIL import Image
from keras.optimizers import SGD  

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

def get_files(filename):
    
     
     label_train = []
     image_train=[]
     for train_class in os.listdir(filename):
         for pic in os.listdir(filename+train_class):
             img = Image.open(filename+train_class+'/'+pic)
             new_img = img.resize((224, 224), Image.BILINEAR)
             image=np.array(new_img)
             image_train.append(image)
             label_train.append(train_class)
     image_train=np.array(image_train)
     label_train=np.array(label_train)
    
     return image_train,label_train#由此可见，但结果数据量很大时，输出列表，结
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    the identity_block is the block that has no conv layer at shortcut
    params:
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    dim_ordering = K.image_dim_ordering()
    nb_filter1, nb_filter2, nb_filter3 = filters
    if dim_ordering == 'tf':
        axis = 3
    else:
        axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    out = Convolution2D(nb_filter1, 1, 1, dim_ordering=dim_ordering, name=conv_name_base + '2a')(input_tensor)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2a')(out)
    out = Activation('relu')(out)

    out = out = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                              dim_ordering=dim_ordering, name=conv_name_base + '2b')(out)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2b')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter3, 1, 1, dim_ordering=dim_ordering, name=conv_name_base + '2c')(out)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2c')(out)

    out = merge([out, input_tensor], mode='sum')
    out = Activation('relu')(out)
    return out


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    conv_block is the block that has a conv layer at shortcut
    params:
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should has subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        axis = 3
    else:
        axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    out = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                        dim_ordering=dim_ordering, name=conv_name_base + '2a')(input_tensor)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2a')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                        dim_ordering=dim_ordering, name=conv_name_base + '2b')(out)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2b')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter3, 1, 1, dim_ordering=dim_ordering, name=conv_name_base + '2c')(out)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2c')(out)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             dim_ordering=dim_ordering, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=axis, name=bn_name_base + '1')(shortcut)

    out = merge([out, shortcut], mode='sum')
    out = Activation('relu')(out)
    return out





def get_resnet50():
    """
    this function returns the 50-layer residual network model
    you should load pretrained weights if you want to use it directly.
    Note that since the pretrained weights is converted from caffemodel
    the order of channels for input image should be 'BGR' (the channel order of caffe)
    """
    if K.image_dim_ordering() == 'tf':
        inp = Input(shape=(224, 224, 3))
        axis = 3
    else:
        inp = Input(shape=(3, 224, 224))
        axis = 1

    dim_ordering = K.image_dim_ordering()
    out = ZeroPadding2D((3, 3), dim_ordering=dim_ordering)(inp)
    out = Convolution2D(64, 7, 7, subsample=(2, 2), dim_ordering=dim_ordering, name='conv1')(out)
    out = BatchNormalization(axis=axis, name='bn_conv1')(out)
    out = Activation('relu')(out)
    out = MaxPooling2D((3, 3), strides=(2, 2), dim_ordering=dim_ordering)(out)

    out = conv_block(out, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    out = identity_block(out, 3, [64, 64, 256], stage=2, block='b')
    out = identity_block(out, 3, [64, 64, 256], stage=2, block='c')

    out = conv_block(out, 3, [128, 128, 512], stage=3, block='a')
    out = identity_block(out, 3, [128, 128, 512], stage=3, block='b')
    out = identity_block(out, 3, [128, 128, 512], stage=3, block='c')
    out = identity_block(out, 3, [128, 128, 512], stage=3, block='d')

    out = conv_block(out, 3, [256, 256, 1024], stage=4, block='a')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='b')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='c')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='d')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='e')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='f')

    out = conv_block(out, 3, [512, 512, 2048], stage=5, block='a')
    out = identity_block(out, 3, [512, 512, 2048], stage=5, block='b')
    out = identity_block(out, 3, [512, 512, 2048], stage=5, block='c')

    out = AveragePooling2D((7, 7), dim_ordering=dim_ordering)(out)
    out = Flatten()(out)
    out = Dense(6, activation='softmax', name='fc1000')(out)

    model = Model(inp, out)

    return model

if __name__ == '__main__':
    data ="/home/lab326/songpeng/dataset/"
    image,label = get_files(data)
    X_train,X_test,Y_train,Y_test=train_test_split(image,label,test_size = 0.2)
    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)
    
    # 将特征点从0~255转换成0~1提高特征提取精度
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    model=get_resnet50()
    sgd = SGD(decay=0.0001,momentum=0.9)  
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    
    model.fit(X_train, Y_train, batch_size=60, epochs=40)
    model.save_weights("my_model_weights.h5") 
      
    score = model.evaluate(X_test, Y_test, batch_size=60)
    print(score)

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    