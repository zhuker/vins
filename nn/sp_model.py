from keras.layers import Input,Reshape, Flatten, Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, GRU, Activation, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from spatialTransformer import SpatialTransformer
import keras.backend as K

K.set_learning_phase(1)

def sp_model(shape, vocab_size, returnTestFunctions = False):

    pool_size = 2
    rnn_size = 512
    max_string_len = 17

    transformedInputSize = (32, max_string_len * 32, 1)

    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((256, 6), dtype='float32')
    weights = [W, b.flatten()]

    inp = Input(shape, name='the_input')
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
    xM = Dropout(0.25)(x)
    x = Dense(256)(xM)
    x = LeakyReLU(0.2)(x)
    predictedpoints = Dense(5, activation='tanh', name='points')(x)

    pointsModel = Model(inputs=inp, outputs=predictedpoints)
    pointsModel.load_weights('checkpoints/vl0.0026.hdf5')

    for layer in pointsModel.layers[:-4]:
        layer.trainable = False

    x = Dense(256, name='m_embeding1')(xM)
    x = LeakyReLU(0.2)(x)
    x = Dense(6, weights=weights, name='locnet_output')(x)
    locnet = Model(input=inp, output=x)
    for layer in locnet.layers[:-4]:
        layer.trainable = False

    transformedInput = SpatialTransformer(localization_net=locnet, output_size=transformedInputSize[:2],
                                          input_shape=shape)(inp)
    inner = Conv2D(64, 3, padding='same',
                   activation='relu', kernel_initializer='he_normal',
                   name='conv1')(transformedInput)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(128, 3, padding='same',
                   activation='relu', kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
    inner = Conv2D(128, 3, padding='same',
                   activation='relu', kernel_initializer='he_normal',
                   name='conv3')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max3')(inner)
    inner = Conv2D(256, 3, padding='same',
                   activation='relu', kernel_initializer='he_normal',
                   name='conv4')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max4')(inner)

    conv_to_rnn_dims = (transformedInputSize[1] // (pool_size ** 4), (transformedInputSize[0] // (pool_size ** 4)) * 256)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(vocab_size, activation='relu', name='dense1')(inner)

    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(vocab_size, kernel_initializer='he_normal', name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=inp, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer

    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[inp, labels, input_length, label_length], outputs=[loss_out,predictedpoints])

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred, 'points':'mse'}, optimizer=sgd, loss_weights=[1, 1])

    test_transformer = K.function([inp], [transformedInput])
    test_all = K.function([inp], [y_pred])
    if not returnTestFunctions:
        return model
    else:
        return model, test_transformer, test_all

# im_heigth = 9*40
# im_width = 16*40
# input_shape = (im_heigth, im_width, 1)
# sp_model(input_shape, 42)