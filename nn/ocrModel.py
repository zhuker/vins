from keras.layers import Input,Reshape, Flatten, Dense, Deconv2D, Conv2D,Conv1D, LSTM, Bidirectional, Dropout, BatchNormalization, MaxPooling2D, GRU, Activation, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from spatialTransformer import SpatialTransformer
import keras.backend as K
from utils import vocabulary
from keras.layers.wrappers import TimeDistributed

import os
currPath = os.path.dirname(os.path.abspath(__file__))

def getOCRModel():

    inp = Input((32,32*16,1))

    c1 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(64, kernel_size=3,  activation='linear', padding='same')(inp)))
    c1 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, activation='linear', padding='same')(c1)))
    m1 = MaxPooling2D((2,2))(c1) #16x(16*16)

    c2 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(96, kernel_size=3,  activation='linear', padding='same')(m1)))
    c2 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(96, kernel_size=3, activation='linear', padding='same')(c2)))
    m2 = MaxPooling2D((2,2))(c2) #8x(8*16)

    c3 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(128, kernel_size=3,  activation='linear', padding='same')(m2)))
    c3 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(128, kernel_size=3, activation='linear', padding='same')(c3)))
    m3 = MaxPooling2D((2,2))(c3) #4x(4*16)

    c4 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(256, kernel_size=3,  activation='linear', padding='same')(m3)))
    c4 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(256, kernel_size=3, activation='linear', padding='same')(c4)))
    m4 = MaxPooling2D((2,2))(c4) #2x(2*16)

    c5 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(256, kernel_size=3, activation='linear', padding='same')(m4)))
    dec = LeakyReLU(alpha=0.1)(BatchNormalization()(Deconv2D(256, kernel_size=(1,3), activation='linear', padding='valid')(c5)))
    m5 = MaxPooling2D((2, 2))(dec) #1x(1*17)

    resh = Reshape((17,256))(m5)

    lastConv =  LeakyReLU(alpha=0.1)(BatchNormalization()(Conv1D(256, kernel_size=3,  activation='linear', padding='same')(resh)))
    drop = Dropout(0.5)(lastConv)
    lstm = Bidirectional(LSTM(128, return_sequences=True))(drop)
    drop2 = Dropout(0.25)(lstm)
    out = TimeDistributed(Dense(len(vocabulary), activation='softmax'))(drop2)
    # flat = Flatten()(lastConv)
    # out = Dense(17*len(vocabulary), activation='linear')(flat)
    # outReshape = Reshape((17,len(vocabulary)))(out)

    model = Model(input=inp, output=out)
    model.compile('adam', 'categorical_crossentropy')
    model.summary()
    model.load_weights(currPath+'/checkpoints/OCRmodel_vl0.7638.hdf5')

    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')
    return model


#getOCRModel()