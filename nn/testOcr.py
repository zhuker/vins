from segmentmodel import getSegModel
from utils import vocabulary
from PIL import Image
import numpy as np
import cv2
import os
from scipy.misc import imsave
from adjustTextPosition import adjust
from ocrModel import getOCRModel
from PIL import Image, ImageFont, ImageDraw, ImageOps

im_heigth = 8*40
im_width = 16*40
input_shape = (im_heigth, im_width, 1)
max_angle = 45

maskModel = getSegModel(input_shape)
maskModel.load_weights('checkpoints/segmenter_vl0.0163.hdf5')

ocrModel = getOCRModel()
ocrModel.load_weights('checkpoints/OCRmodel_vl0.7638.hdf5')

imgs = [os.path.join(root, f) for root, _, files in os.walk('./tmp1') for f in files if f.lower().endswith('.jpg')]


def readLabel(label):
    vin = ''
    for charprobs in label:
        charid = np.argmax(charprobs)
        vin += vocabulary[charid]
    return vin

for imgpath in imgs:
    savepath = imgpath.replace('tmp1', 'finalResults')
    img = Image.open(imgpath)
    img = img.resize((im_width, im_heigth))

    img = np.array(img.convert('L'))
    np_img = img / 127.5 - 1
    np_img = np.reshape(np_img, (1, im_heigth, im_width, 1))

    mask = maskModel.predict(np_img)[0]
    mask *= 255
    mask = mask.astype(np.uint8)[:, :, 0]


    res, newmask = adjust(img, mask)

    cropVin = np.expand_dims(np.array([cv2.resize(res,(32*16, 32)).astype(np.float)/127.5 -1]),axis=-1)
    label = ocrModel.predict(cropVin)[0]
    vin = readLabel(label)

    p = savepath.split('/')[:-1]
    p.append(vin + '.jpg')
    savepath = '/'.join(p)

    text_image = Image.new('L', (cropVin.shape[2], cropVin.shape[1]*2))
    draw = ImageDraw.Draw(text_image)
    font = ImageFont.truetype('fonts/roboto/Roboto-Regular.ttf', 32)
    draw.text((0, 0), vin, font=font, fill=255)

    trueVin = imgpath.split('/')[-1].split('_')[0]
    draw.text((0, 32), trueVin, font=font, fill=200)
    res = np.vstack((cv2.resize(res, (cropVin.shape[2], cropVin.shape[1])), np.array(text_image)))


    imsave(savepath, res)
    print(savepath)

