from keras.layers import Input, Flatten, Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU


def bbox_model(shape, coords_count):

    inp = Input(shape)
    x = Conv2D(64, (4, 4), padding='valid', strides=2)(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, (4, 4), padding='valid', strides=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (3, 3), padding='same', strides=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (4, 4), padding='valid', strides=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (4, 4), padding='valid', strides=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (3, 3), padding='same', strides=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (4, 4), padding='valid', strides=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (3, 3), padding='same', strides=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(coords_count, activation='tanh')(x)

    model = Model(inputs=inp, outputs=x)

    return model
