import numpy as np
import cv2
import os
from scipy.misc import imsave
from bboxModel import bboxModel
from PIL import Image, ImageFont, ImageDraw
from utils import drawBBox

im_height = 8 * 20
im_width = 16 * 20
input_shape = (im_height, im_width, 1)

model = bboxModel()
model.summary()
model.load_weights('checkpoints/BBoxModel_vl0.3347.hdf5')

# ocr_model = timecode_ocr_model_small()
# ocr_model.load_weights('checkpoints/OCRmodel_vl0.0424.hdf5')

imgs = [os.path.join(root, f) for root, _, files in os.walk('real/') for f in files if f.lower().endswith('.jpg')]
imgs.sort()

vocabulary = '0123456789:;'


def readLabel(label):
    vin = ''
    for charprobs in label:
        charid = np.argmax(charprobs)
        vin += vocabulary[charid]
    return vin


for imgpath in imgs:
    savepath = imgpath.replace('real', 'res')
    img = Image.open(imgpath)

    img = cv2.resize(np.array(img.convert('L')),(im_width, im_height)).reshape((im_height, im_width, 1))
    np_img = img / 127.5 - 1
    np_img = np.reshape(np_img, (1, im_height, im_width, 1))

    bbox = model.predict(np_img)[0][0]

    drawBBox(bbox, img)
    cv2.imwrite(savepath, img)
    print(savepath)

    # sudo fuser -v /dev/nvidia*
