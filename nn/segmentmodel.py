from keras.layers import Deconv2D, Conv2D, MaxPooling2D
from keras.models import Sequential


def getSegModel(input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3,  activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(128, kernel_size=3,  activation='relu', padding='same'))
    model.add(Deconv2D(128, kernel_size=4, strides=2, activation='relu', padding='same'))
    model.add(Deconv2D(128, kernel_size=4, strides=2, activation='relu', padding='same'))
    model.add(Deconv2D(64, kernel_size=4, strides=2, activation='relu', padding='same'))
    model.add(Conv2D(1, kernel_size=3,  activation='sigmoid', padding='same'))
    return model
