from segmentmodel import getSegModel
from timecode_ocr_model import timecode_ocr_model
from PIL import Image
import numpy as np
from adjustTextPosition import adjust
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageOps
import os
currPath = os.path.dirname(os.path.abspath(__file__))

vocabulary = '0123456789:;'

def readLabel(label):
    vin = ''
    for charprobs in label:
        charid = np.argmax(charprobs)
        vin += vocabulary[charid]
    return vin

class MaskOcrModel(object):
    def __init__(self):
        self.im_heigth = 8 * 40
        self.im_width = 16 * 40
        input_shape = (self.im_heigth, self.im_width, 1)
        self.maskModel = getSegModel(input_shape)
        self.ocrModel = timecode_ocr_model()
        self.ocrModel.load_weights(currPath+'/checkpoints/OCRmodel_vl0.0313.hdf5')

    def readText(self, imgpath):
        savepath = imgpath.replace('input', 'output')
        img = Image.open(imgpath)
        img = img.resize((self.im_width, self.im_heigth))

        img = np.array(img.convert('L'))
        np_img = img / 127.5 - 1
        np_img = np.reshape(np_img, (1, self.im_heigth, self.im_width, 1))

        mask = self.maskModel.predict(np_img)[0]
        mask *= 255
        mask = mask.astype(np.uint8)[:, :, 0]

        res, newmask = adjust(img, mask)

        cropVin = np.expand_dims(np.array([cv2.resize(res, (32 * 16, 32)).astype(np.float) / 127.5 - 1]), axis=-1)
        label = self.ocrModel.predict(cropVin)[0]
        vin = readLabel(label)

        savepath = savepath.replace('.jpg','_'+vin+'.jpg')

        text_image = Image.new('L', (cropVin.shape[2], cropVin.shape[1]))
        draw = ImageDraw.Draw(text_image)
        font = ImageFont.truetype(currPath + '/fonts/roboto/Roboto-Regular.ttf', 32)
        draw.text((0, 0), vin, font=font, fill=255)
        res = np.vstack((cv2.resize(res, (cropVin.shape[2], cropVin.shape[1])), np.array(text_image)))

        cv2.imwrite(savepath, res)

        res = {'imgpath':savepath.replace('/static/','/static/'),
               'text':vin}
        return res