from keras.layers import Input, Reshape, Dense, Deconv2D, Conv2D, Conv1D, LSTM, Bidirectional, Dropout, BatchNormalization, MaxPooling2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.wrappers import TimeDistributed


def timecode_ocr_model():

    inp = Input((32, 32*16, 1))

    c1 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(64, kernel_size=3, padding='same')(inp)))
    c1 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, padding='same')(c1)))
    m1 = MaxPooling2D((2, 2))(c1)

    c2 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(96, kernel_size=3, padding='same')(m1)))
    c2 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(96, kernel_size=3, padding='same')(c2)))
    m2 = MaxPooling2D((2, 2))(c2)

    c3 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(128, kernel_size=3, padding='same')(m2)))
    c3 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(128, kernel_size=3, padding='same')(c3)))
    m3 = MaxPooling2D((2, 2))(c3)

    c4 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(256, kernel_size=3, padding='same')(m3)))
    c4 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(256, kernel_size=3, padding='same')(c4)))
    m4 = MaxPooling2D((2, 2))(c4)

    c5 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(256, kernel_size=3, padding='same')(m4)))
    d1 = LeakyReLU(alpha=0.1)(BatchNormalization()(Deconv2D(256, kernel_size=(1, 7), padding='valid')(c5)))
    d2 = LeakyReLU(alpha=0.1)(BatchNormalization()(Deconv2D(256, kernel_size=(1, 7), padding='valid')(d1)))
    m6 = MaxPooling2D((1, 2))(d2)

    c7 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(256, kernel_size=3, padding='same')(m6)))
    m7 = MaxPooling2D((2, 2))(c7)
    resh = Reshape((11, 256))(m7)
    last_conv = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv1D(256, kernel_size=3, padding='same')(resh)))

    drop = Dropout(0.5)(last_conv)
    lstm = Bidirectional(LSTM(128, return_sequences=True))(drop)
    drop2 = Dropout(0.25)(lstm)
    out = TimeDistributed(Dense(12, activation='softmax'))(drop2)

    model = Model(input=inp, output=out)
    model.compile('adam', 'categorical_crossentropy')
    # model.summary()

    return model


# timecode_ocr_model().summary()
