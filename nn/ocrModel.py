from keras.layers import Input,Reshape, Flatten, Dense, Deconv2D, Conv2D,Conv1D, LSTM, Bidirectional, Dropout, BatchNormalization, MaxPooling2D, GRU, Activation, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.wrappers import TimeDistributed


def getOCRModel():

    inp = Input((32, 32*16, 1))

    c1 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(64, kernel_size=3, padding='same')(inp)))
    c1 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, padding='same')(c1)))
    m1 = MaxPooling2D((2, 2))(c1) #16x(16*16)

    c2 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(96, kernel_size=3, padding='same')(m1)))
    c2 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(96, kernel_size=3, padding='same')(c2)))
    m2 = MaxPooling2D((2, 2))(c2) #8x(8*16)

    c3 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(128, kernel_size=3, padding='same')(m2)))
    c3 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(128, kernel_size=3, padding='same')(c3)))
    m3 = MaxPooling2D((2, 2))(c3) #4x(4*16)

    c4 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(256, kernel_size=3, padding='same')(m3)))
    c4 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(256, kernel_size=3, padding='same')(c4)))
    m4 = MaxPooling2D((2, 2))(c4) #2x(2*16)

    c5 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(256, kernel_size=3, padding='same')(m4)))
    dec = LeakyReLU(alpha=0.1)(BatchNormalization()(Deconv2D(256, kernel_size=(1, 3), padding='valid')(c5)))
    m5 = MaxPooling2D((2, 2))(dec) #1x(1*17)

    resh = Reshape((17, 256))(m5)

    lastConv = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv1D(256, kernel_size=3, padding='same')(resh)))
    drop = Dropout(0.5)(lastConv)
    lstm = Bidirectional(LSTM(128, return_sequences=True))(drop)
    drop2 = Dropout(0.25)(lstm)
    out = TimeDistributed(Dense(17, activation='softmax'))(drop2)


    model = Model(input=inp, output=out)
    model.compile('adam', 'categorical_crossentropy')
    model.summary()


    return model


getOCRModel()