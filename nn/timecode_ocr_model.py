from keras.layers import Input, Reshape,RepeatVector,Flatten, Dense, Deconv2D, Activation,concatenate,Convolution2D,Conv2D, Conv1D, LSTM, Bidirectional, Dropout, BatchNormalization, MaxPooling2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.wrappers import TimeDistributed
import keras.backend as K

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



def timecode_ocr_model_small():

    inp = Input((32, 32*11, 1))

    c1 = LeakyReLU(alpha=0.1)(BatchNormalization()(Convolution2D(32, kernel_size=3, padding='same')(inp)))
    c1 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(32, kernel_size=3, padding='same')(c1)))
    m1 = MaxPooling2D((2, 2))(c1)

    c2 = LeakyReLU(alpha=0.1)(BatchNormalization()(Convolution2D(48, kernel_size=3, padding='same')(m1)))
    c2 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(48, kernel_size=3, padding='same')(c2)))
    m2 = MaxPooling2D((2, 2))(c2)

    c3 = LeakyReLU(alpha=0.1)(BatchNormalization()(Convolution2D(64, kernel_size=3, padding='same')(m2)))
    c3 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(64, kernel_size=3, padding='same')(c3)))
    m3 = MaxPooling2D((2, 2))(c3)

    c4 = LeakyReLU(alpha=0.1)(BatchNormalization()(Convolution2D(96, kernel_size=3, padding='same')(m3)))
    c4 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(96, kernel_size=3, padding='same')(c4)))
    m4 = MaxPooling2D((2, 2))(c4)

    c5 = LeakyReLU(alpha=0.1)(BatchNormalization()(Convolution2D(128, kernel_size=3, padding='same')(m4)))
    m6 = MaxPooling2D((2, 2))(c5)

    c7 = LeakyReLU(alpha=0.1)(BatchNormalization()(Convolution2D(128, kernel_size=3, padding='same')(m6)))
    resh = Reshape((11, 128))(c7)
    drop = Dropout(0.25)(resh)
    lstm1 = LSTM(64, return_sequences=True)(drop)
    drop2 = Dropout(0.25)(concatenate([lstm1, resh], axis=-1))
    resh2 = Reshape((1, 11, 128+64))(drop2)
    out = Convolution2D(12, kernel_size=1, padding='same')(resh2)
    out = Reshape((11, 12))(out)
    out = Activation('softmax')(out)

    model = Model(input=inp, output=out)
    model.compile('adam', 'categorical_crossentropy')
    # model.summary()

    return model

def timecode_ocr_model_RGB():

    inp = Input((32, 32*16, 3))

    c1 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(32, kernel_size=3, padding='valid')(inp)))
    m1 = MaxPooling2D((2, 2))(c1)

    c2 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(48, kernel_size=3, padding='valid')(m1)))
    m2 = MaxPooling2D((1, 2))(c2)

    c3 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(64, kernel_size=3, padding='valid')(m2)))
    m3 = MaxPooling2D((1, 2))(c3)

    c4 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(96, kernel_size=3, padding='valid')(m3)))

    c5 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(128, kernel_size=(5,3), padding='valid')(c4)))

    c7 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(128, kernel_size=(5,3), padding='valid')(c5)))
    resh = Reshape((56, 128))(c7)
    drop = Dropout(0.25)(resh)
    emb = Bidirectional(LSTM(64, return_sequences=False))(drop)
    dec = Bidirectional(LSTM(64, return_sequences=True))(RepeatVector(11)(emb))
    resh2 = Reshape((1, 11, 128))(dec)
    out = Convolution2D(12, kernel_size=1, padding='same')(resh2)
    out = Reshape((11, 12))(out)
    out = Activation('softmax')(out)

    model = Model(input=inp, output=out)
    model.compile('adam', 'categorical_crossentropy')
    # model.summary()

    return model



def timecode_ocr_model_RGB_full():

    inp = Input((27, 27*7, 3))

    c1 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(64, kernel_size=3, padding='same', trainable=False)(inp)))
    c1 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(64, kernel_size=3, padding='same', trainable=False)(c1)))
    m1 = MaxPooling2D((2, 2), trainable=False)(c1)

    c2 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(96, kernel_size=3, padding='same', trainable=False)(m1)))
    c2 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(96, kernel_size=3, padding='same', trainable=False)(c2)))
    m2 = MaxPooling2D((2, 2), trainable=False)(c2)

    c3 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(128, kernel_size=3, padding='same', trainable=False)(m2)))
    m3 = MaxPooling2D((2, 2), trainable=False)(c3)

    c4 = LeakyReLU(alpha=0.2)(BatchNormalization()(Convolution2D(128, kernel_size=(3,1), padding='valid', trainable=False)(m3)))

    resh = Reshape((23, 128), trainable=False)(c4)
    drop = Dropout(0.25)(resh)
    emb = Bidirectional(LSTM(64, return_sequences=False, trainable=False))(drop)
    dec = Bidirectional(LSTM(64, return_sequences=True, trainable=False))(RepeatVector(11, trainable=False)(emb))
    resh2 = Reshape((1, 11, 128), trainable=False)(dec)
    out = Convolution2D(12, kernel_size=1, padding='same', trainable=False)(resh2)
    out = Reshape((11, 12), trainable=False)(out)
    out = Activation('softmax', name='chars', trainable=False)(out)

    emb_istimecode = Dense(64, activation='relu')(emb)
    isTimecode = Dense(1, activation='sigmoid', name='conf')(emb_istimecode)


    model = Model(input=inp, output=[out, isTimecode])
    model.compile('adam', ['categorical_crossentropy', 'binary_crossentropy'], loss_weights=[0.,1.])
    model.summary()

    return model


# m = timecode_ocr_model_RGB_full()
#
# m.load_weights('checkpoints/timecode_ocr_model_RGB_full_vl0.0328.hdf5')
# m.save('/home/oles/Vin_ocr/vins/JS/weights/OCRmodel.h5')
# m.save_weights('/home/oles/Vin_ocr/vins/JS/weights/OCRmodel_weights.hdf5')
# with open('/home/oles/Vin_ocr/vins/JS/weights/OCRmodel.json', 'w') as f:
#     f.write(m.to_json())
