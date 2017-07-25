from segmentmodel import getSegModel
from utils import vocabulary
from PIL import Image
import numpy as np
from scipy.misc import imsave
from utils import crop_rotate, mirror
import cv2
import os
import math
import itertools
from scipy.misc import imsave

im_heigth = 8*40
im_width = 16*40
input_shape = (im_heigth, im_width, 1)
max_angle = 45

model = getSegModel(input_shape)
model.load_weights('checkpoints/segmenter_vl0.0258.hdf5')

imgs = [os.path.join(root, f) for root, _, files in os.walk('./tmp') for f in files if f.lower().endswith('.jpg')]


for imgpath in imgs:
    img = Image.open(imgpath)
    img = img.resize((im_width, im_heigth))

    img = np.array(img.convert('L'))
    np_img = img / 127.5 - 1
    np_img = np.reshape(np_img, (1, im_heigth, im_width, 1))

    mask = model.predict(np_img)[0]
    mask *= 255

    result = np.vstack((mask.astype(np.uint8)[:, :, 0], img))
    imsave(imgpath.replace('tmp', 'res'), result)

    # sudo fuser -v /dev/nvidia*

