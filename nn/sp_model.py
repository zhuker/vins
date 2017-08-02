from keras.layers import Input,Reshape, Flatten, Dense, Cropping2D, Conv2D, Deconv2D, Conv1D, TimeDistributed, Bidirectional, LSTM, Dropout, BatchNormalization, MaxPooling2D, GRU, Activation, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from spatialTransformer import SpatialTransformer
import keras.backend as K
from utils import vocabulary
from segmentmodel import getSegModel

K.set_learning_phase(1)

im_heigth = 8 * 40
im_width = 16 * 40
input_shape = (im_heigth, im_width, 1)

def OcrModel():
    inp = Input((32,32*16,1))

    c1 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(64, kernel_size=3, activation='linear', padding='same')(inp)))
    c1 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, activation='linear', padding='same')(c1)))
    m1 = MaxPooling2D((2, 2))(c1)  # 16x(16*16)

    c2 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(96, kernel_size=3, activation='linear', padding='same')(m1)))
    c2 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(96, kernel_size=3, activation='linear', padding='same')(c2)))
    m2 = MaxPooling2D((2, 2))(c2)  # 8x(8*16)

    c3 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(128, kernel_size=3, activation='linear', padding='same')(m2)))
    c3 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(128, kernel_size=3, activation='linear', padding='same')(c3)))
    m3 = MaxPooling2D((2, 2))(c3)  # 4x(4*16)

    c4 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(256, kernel_size=3, activation='linear', padding='same')(m3)))
    c4 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(256, kernel_size=3, activation='linear', padding='same')(c4)))
    m4 = MaxPooling2D((2, 2))(c4)  # 2x(2*16)

    c5 = LeakyReLU(alpha=0.1)(BatchNormalization()(Conv2D(256, kernel_size=3, activation='linear', padding='same')(m4)))
    dec = LeakyReLU(alpha=0.1)(
        BatchNormalization()(Deconv2D(256, kernel_size=(1, 3), activation='linear', padding='valid')(c5)))
    m5 = MaxPooling2D((2, 2))(dec)  # 1x(1*17)

    resh = Reshape((17, 256))(m5)

    lastConv = LeakyReLU(alpha=0.1)(
        BatchNormalization()(Conv1D(256, kernel_size=3, activation='linear', padding='same')(resh)))
    drop = Dropout(0.5)(lastConv)
    lstm = Bidirectional(LSTM(128, return_sequences=True))(drop)
    drop2 = Dropout(0.25)(lstm)
    out = TimeDistributed(Dense(len(vocabulary), activation='softmax'))(drop2)


    model = Model(input=inp, output=out)
    model.load_weights('checkpoints/OCRmodel_vl0.7638.hdf5')

    for layer in model.layers:
        layer.trainable = False

    return model


def locnetWithMask(input_shape):
    segmodel = getSegModel(input_shape)
    for layer in segmodel.layers:
        layer.trainable = False
    inp = Input(input_shape)
    segmentMask = segmodel(inp)
    combinedinput = concatenate([inp,segmentMask], axis=-1)

    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, padding='same')(combinedinput))))
    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, padding='same')(x))))
    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(128, kernel_size=3, padding='same')(x))))
    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(128, kernel_size=3, padding='same')(x))))
    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(256, kernel_size=3, padding='same')(x))))
    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(256, kernel_size=3, padding='same')(x))))
    xZ = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(256, kernel_size=3, padding='same')(x))))
    x = Flatten()(xZ)
    x = Dense(256, name='emb1', activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(256, name='emb2', activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(6, name='out')(x)
    locnet = Model(input=inp, output=x)

    locnet.load_weights('checkpoints/LOCNET_vl0.0469.hdf5')
    locnet.compile('adam', 'mse')
    locnet.summary()
    return  locnet

def getlocnet(input_shape):

    inp = Input(input_shape, name='the_input')
    x = MaxPooling2D((2, 2))(
        LeakyReLU(0.2)(BatchNormalization()(Conv2D(32, kernel_size=3, padding='same')(inp))))
    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(32, kernel_size=3, padding='same')(x))))
    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, padding='same')(x))))
    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, padding='same')(x))))
    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(128, kernel_size=3, padding='same')(x))))
    x = MaxPooling2D((1, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(128, kernel_size=3, padding='same')(x))))
    xZ = LeakyReLU(0.2)(BatchNormalization()(Conv2D(128, kernel_size=3, padding='same')(x)))
    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(64, kernel_size=3,  padding='same')(xZ))))
    x = Reshape((64*5*5,))(x)
    x = Dropout(0.25)(x)
    x = LeakyReLU(0.2)(Dense(256, name='emb2')(x))
    x = Dropout(0.25)(x)
    x = Dense(6,activation='linear', name='out')(x)
    #x = Lambda(lambda y: y * 2)(x)

    locnet = Model(input=inp, output=x)

    #locnet.load_weights('checkpoints/LOCNET_vl0.0574.hdf5')
    locnet.compile('adam', 'mse')
    # for layer in locnet.layers[:-3]:
    #     layer.trainable = False
    return locnet

def locnet_and_sp_mask(input_shape):
    segmodel = getSegModel(input_shape, compile=False)
    transformedInputSize = (32, 16 * 32, 1)

    # for layer in segmodel.layers:
    #     layer.trainable = False
    inp = Input(input_shape)
    segmentMask = segmodel(inp)
    combinedinput = concatenate([inp, segmentMask], axis=-1)

    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(32, kernel_size=3, padding='same')(segmentMask))))
    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(32, kernel_size=3, padding='same')(x))))
    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, padding='same')(x))))
    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, padding='same')(x))))
    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(128, kernel_size=3, padding='same')(x))))
    x = MaxPooling2D((2, 2))(LeakyReLU(0.2)(BatchNormalization()(Conv2D(128, kernel_size=3, padding='same')(x))))
    xZ = LeakyReLU(0.2)(BatchNormalization()(Conv2D(128, kernel_size=3, padding='same')(x)))
    x = LeakyReLU(0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, padding='same')(xZ)))
    x = Reshape((2048,))(x)
    x = Dropout(0.25)(x)
    x = LeakyReLU(0.2)(Dense(256, name='emb2')(x))
    x = Dropout(0.25)(x)
    x = Dense(6, name='out')(x)
    locnet = Model(input=inp, output=x)
    #locnet.compile('adam','mse')
    #locnet.load_weights('checkpoints/LOCNET_vl0.0469.hdf5')

    transformedMask = SpatialTransformer(localization_net=locnet, output_size=transformedInputSize[:2], input_shape=input_shape)(segmentMask)
    model = Model(input = inp, outputs=[x, transformedMask, segmentMask])

    model.compile('sgd', ['mse', 'mse', 'binary_crossentropy'], loss_weights=[1.,0.,1.])
    model.summary()
    return model

def sp_model(shape, retunCrop=False):

    ocrModel = OcrModel()


    transformedInputSize = (32, 16 * 32, 1)

    inp = Input(shape)
    locnet_output = locnetWithMask(shape)
    transformedInput = SpatialTransformer(localization_net=locnet_output, output_size=transformedInputSize[:2],
                                          input_shape=shape)(inp)
    ocrOutput = ocrModel(transformedInput)
    if retunCrop:
        endToEndModel = Model(input= inp, output=[ocrOutput, transformedInput])
    else:
        endToEndModel = Model(input = inp, output= ocrOutput)
    sgd = SGD(lr=0.001)
    endToEndModel.compile(sgd, 'categorical_crossentropy')
    return endToEndModel

def locnet_on_mask(shape, retunCrop=False):


    transformedInputSize = (32, 16 * 32, 1)

    locnet = getlocnet(shape)

    inp  = Input(shape)

    transformedInput = SpatialTransformer(localization_net=locnet,
                                          output_size=transformedInputSize[:2],
                                          input_shape=shape)(inp)

    segmodel = getSegModel(transformedInputSize)
    for layer in segmodel.layers:
        layer.trainable = False
    segOutput = segmodel(transformedInput)

    if retunCrop:
        endToEndModel = Model(input=inp, output=[segOutput, transformedInput])
    else:
        endToEndModel = Model(input=inp, output=segOutput)
    endToEndModel.compile('sgd', 'binary_crossentropy')
    return endToEndModel

def testMatrixModel():
    transformedInputSize = (64, 16 * 32, 1)
    im_heigth = 9 * 40
    im_width = 16 * 40
    input_shape = (im_heigth, im_width, 1)

    imgInp = Input(input_shape, name='imageInput')
    x = Cropping2D(cropping=((0, im_heigth-1), (0, im_width-6)))(imgInp)
    matrix = Flatten()(x)
    locnet = Model(inputs = imgInp, output = matrix)


    sp_Imginput  = Input(input_shape)
    transformedInput = SpatialTransformer(localization_net=locnet,
                                          output_size=transformedInputSize[:2],
                                          input_shape=input_shape)(sp_Imginput)
    output = Reshape(transformedInputSize)(transformedInput)
    model = Model(inputs=sp_Imginput, output=output)
    model.compile('sgd','mse')
    return model