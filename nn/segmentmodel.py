from keras.layers import Input,Reshape, Flatten, UpSampling2D, Dense,Deconv2D, Conv2D, Dropout, BatchNormalization, MaxPooling2D, GRU, Activation, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from spatialTransformer import SpatialTransformer
import keras.backend as K

import os
currPath = os.path.dirname(os.path.abspath(__file__))

def getSegModel(input_shape, compile=True):
    inp = Input(input_shape)

    c1 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(32, kernel_size=3,  activation='linear', padding='same')(inp)))
    m1 = MaxPooling2D((2,2))(c1)

    c2 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(64, kernel_size=3,  activation='linear', padding='same')(m1)))
    m2 = MaxPooling2D((2,2))(c2)

    c3 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(64, kernel_size=3,  activation='linear', padding='same')(m2)))
    m3 = MaxPooling2D((2,2))(c3)

    c4 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(64, kernel_size=3,  activation='linear', padding='same')(m3)))
    m4 = MaxPooling2D((2,2))(c4)

    c5 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, activation='linear', padding='same')(m4)))
    m5 = MaxPooling2D((2, 2))(c5)

    emb = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, activation='linear', padding='same')(m5)))

    dec5 = LeakyReLU(alpha=0.2)(BatchNormalization()(Deconv2D(64, kernel_size=4, strides=2, padding='same')(emb)))
    concat5 = concatenate([dec5, c5], axis=-1)

    dec4 = LeakyReLU(alpha=0.2)(BatchNormalization()(Deconv2D(64, kernel_size=4, strides=2, padding='same')(concat5)))
    concat4 = concatenate([dec4, c4], axis=-1)

    dec3 = LeakyReLU(alpha=0.2)(BatchNormalization()(Deconv2D(64, kernel_size=4, strides=2, padding='same')(concat4)))
    concat3 = concatenate([dec3, c3], axis=-1)

    dec2 = LeakyReLU(alpha=0.2)(BatchNormalization()(Deconv2D(64, kernel_size=4, strides=2, padding='same')(concat3)))
    concat2 = concatenate([dec2, c2], axis=-1)

    dec1 = LeakyReLU(alpha=0.2)(BatchNormalization()(Deconv2D(32, kernel_size=4, strides=2, padding='same')(concat2)))
    concat1 = concatenate([dec1, c1], axis=-1)

    out = Conv2D(1, kernel_size=3,  activation='sigmoid', padding='same')(concat1)

    model = Model(input=inp, output=out)
    if compile:
        model.compile('adam', 'binary_crossentropy')
        model.summary()

        # from keras.utils import plot_model
        # plot_model(model, to_file='model.png')
        model.load_weights(currPath+'/checkpoints/tc_segmenter_vl0.0022.hdf5', by_name=True)
    else:
        #model.load_weights('checkpoints/segmenter_vl0.0163.hdf5', by_name=True)
        return model
    return model



def getSegConvModel(input_shape, compile=True):
    inp = Input(input_shape)

    c1 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(32, kernel_size=3,  activation='linear', padding='same')(inp)))
    m1 = MaxPooling2D((2,2))(c1)

    c2 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(32, kernel_size=3,  activation='linear', padding='same')(m1)))
    m2 = MaxPooling2D((2,2))(c2)

    c3 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(32, kernel_size=3,  activation='linear', padding='same')(m2)))
    m3 = MaxPooling2D((2,2))(c3)

    c4 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(32, kernel_size=3,  activation='linear', padding='same')(m3)))
    m4 = MaxPooling2D((2,2))(c4)

    c5 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(32, kernel_size=3,  activation='linear', padding='same')(m4)))
    up4 = UpSampling2D((2,2))(c5)

    concat1 = concatenate([up4, c4], axis=-1)
    d4 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(32, kernel_size=3,  activation='linear', padding='same')(concat1)))
    up3 = UpSampling2D((2,2))(d4)

    concat2 = concatenate([up3, c3], axis=-1)
    d3 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(32, kernel_size=3,  activation='linear', padding='same')(concat2)))
    up2 = UpSampling2D((2,2))(d3)

    concat3 = concatenate([up2, c2], axis=-1)
    d2 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(32, kernel_size=3,  activation='linear', padding='same')(concat3)))
    up1 = UpSampling2D((2,2))(d2)

    out = Conv2D(1, kernel_size=3,  activation='sigmoid', padding='same')(up1)

    model = Model(input=inp, output=out)
    if compile:
        model.compile('adam', 'binary_crossentropy')
        model.summary()

        # from keras.utils import plot_model
        # plot_model(model, to_file='model.png')
        #model.load_weights(currPath+'/checkpoints/segmenter_vl0.0163.hdf5', by_name=True)
    else:
        #model.load_weights('checkpoints/segmenter_vl0.0163.hdf5', by_name=True)
        return model
    return model