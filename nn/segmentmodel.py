from keras.layers import Input,Reshape, Flatten, Dense,Deconv2D, Conv2D, Dropout, BatchNormalization, MaxPooling2D, GRU, Activation, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from spatialTransformer import SpatialTransformer
import keras.backend as K

def getSegModel(input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3,  activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(128, kernel_size=3,  activation='relu', padding='same'))
    model.add(Deconv2D(128, kernel_size=4, strides=2, activation='relu', padding='same'))
    model.add(Deconv2D(128, kernel_size=4, strides=2, activation='relu', padding='same'))
    model.add(Deconv2D(64, kernel_size=4, strides=2, activation='relu', padding='same'))
    model.add(Conv2D(1, kernel_size=3,  activation='sigmoid', padding='same'))
    model.compile('adam', 'binary_crossentropy')
    return model