from keras.layers import Input,Reshape, Flatten, Dense,Deconv2D, Conv2D, Dropout, BatchNormalization, MaxPooling2D, GRU, Activation, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from spatialTransformer import SpatialTransformer
import keras.backend as K

def getSegModel(input_shape):
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=3,  activation='relu', padding='same', input_shape=input_shape))
    # model.add(MaxPooling2D(2,2))
    # model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    # model.add(MaxPooling2D(2,2))
    # model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    # model.add(MaxPooling2D(2,2))
    # model.add(Conv2D(128, kernel_size=3,  activation='relu', padding='same'))
    # model.add(Deconv2D(128, kernel_size=4, strides=2, activation='relu', padding='same'))
    # model.add(Deconv2D(128, kernel_size=4, strides=2, activation='relu', padding='same'))
    # model.add(Deconv2D(64, kernel_size=4, strides=2, activation='relu', padding='same'))
    # model.add(Conv2D(1, kernel_size=3,  activation='sigmoid', padding='same'))
    # model.compile('adam', 'binary_crossentropy')

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
    model.compile('adam', 'binary_crossentropy')
    model.summary()

    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    return model

im_heigth = 8 * 40
im_width = 16 * 40
input_shape = (im_heigth, im_width, 1)
model = getSegModel(input_shape)
