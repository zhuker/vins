import numpy as np
from keras import callbacks
import random
import os
from PIL import Image, ImageFont, ImageDraw, ImageOps
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import Pool, cpu_count
from ocrModel import getOCRModel
from utils import generateVin, rotate_coord, getLabel, vocabulary
import cv2
from random import shuffle

im_heigth = 32
im_width = 32*16
nb_epoch = 100
BATCH_SIZE  = 8

imgs = [os.path.join(root, f) for root, _, files in os.walk('trainimgs/') for f in files if f.endswith('.jpg')]

shuffle(imgs)
trainimgs = imgs[:-400]
testimgs = imgs[-400:]

def gen(batch_size=8, imglist = []):
    k = 0
    imgCount = len(imglist)
    shuffle(imglist)
    def getLabel(imgname):
        vin = imgname.split('_')[0]
        label = np.zeros((17, len(vocabulary)))
        for i, char in enumerate(vin):
            label[i][vocabulary.index(char)] = 1
        return label

    while True:
        x = np.zeros((batch_size, im_heigth, im_width, 1), dtype='float32')
        y = np.zeros((batch_size, 17, len(vocabulary)))
        for i in range(0, batch_size):
            if k>=imgCount:
                shuffle(imglist)
                k = 0
            k = k%imgCount
            x[i] = cv2.imread(imglist[k])[:,:,:1] / 127.5 - 1
            y[i] = getLabel(imglist[k].split('/')[-1])
            k+=1

        yield (x, y)


model = getOCRModel()
# model = bbox_model(shape=input_shape, coords_count=coords_count)
# model = to_multi_gpu(model, 2)

# model.compile(loss="mean_squared_error", optimizer='sgd')
model.load_weights('checkpoints/OCRmodel_vl1.7862.hdf5')
model.summary()

model.fit_generator(generator=gen(batch_size=BATCH_SIZE, imglist = trainimgs),
                    validation_data=gen(batch_size=BATCH_SIZE, imglist = testimgs),
                    steps_per_epoch=3000,
                    validation_steps=100,
                    epochs=nb_epoch,
                    max_q_size=100,
                    callbacks=[
                        callbacks.ModelCheckpoint(
                            'checkpoints/OCRmodel_vl{val_loss:.4f}.hdf5',
                            monitor='val_loss')
                    ])
