from keras.layers import Input,Reshape, Flatten, Dense, Deconv2D, Conv2D,Conv1D, LSTM, Bidirectional,merge, Dropout, BatchNormalization, MaxPooling2D, GRU, Activation, Lambda, RepeatVector
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.wrappers import TimeDistributed
import keras.backend as K

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
    m5 = MaxPooling2D((2, 2))(c5) #1x(1*17)

    resh = Reshape((16, 256))(m5)

    att = Dense(16*11, activation='linear')(Flatten()(resh))
    att = Reshape((11,16))(att)
    att = Activation('softmax')(att)

    def makeAttentionMatrix(att2d):
        att3d = K.expand_dims(att2d)
        att3d = K.repeat_elements(att3d, 256, 3)
        return att3d

    def makeAdjustShape(features2d):
        features3d = K.expand_dims(features2d, 1)
        features3d = K.repeat_elements(features3d, 11, 1)
        return features3d

    def meanFeatures(features3d):
        features2d = K.mean(features3d, axis=2)
        return features2d

    att = Lambda(makeAttentionMatrix)(att)
    attFeatures3d = Lambda(makeAdjustShape)(resh)

    attFeatures = merge([attFeatures3d, att], mode='mul')
    attFeatures = Lambda(meanFeatures)(attFeatures)

    lstm = Bidirectional(LSTM(128, return_sequences=True))(attFeatures)
    drop2 = Dropout(0.25)(lstm)
    out = TimeDistributed(Dense(12, activation='softmax'))(drop2)


    model = Model(input=inp, output=out)
    model.compile('adam', 'categorical_crossentropy')
    model.summary()


    return model
