from sp_model import sp_model, locnet_and_sp_mask
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
from adjustTextPosition import adjust
from PIL import Image, ImageFont, ImageDraw, ImageOps

im_heigth = 8*40
im_width = 16*40
input_shape = (im_heigth, im_width, 1)
max_angle = 45

model = sp_model(input_shape, retunCrop=True)
model.load_weights('checkpoints/endToEnd1.2899.hdf5')

imgs = [os.path.join(root, f) for root, _, files in os.walk('./tmp') for f in files if f.lower().endswith('.jpg')]

def readLabel(label):
    vin = ''
    for charprobs in label:
        charid = np.argmax(charprobs)
        vin += vocabulary[charid]
    return vin

for imgpath in imgs:
    savepath = imgpath.replace('tmp', 'res')
    if not os.path.exists(savepath):
        img = Image.open(imgpath)
        img = img.resize((im_width, im_heigth))

        img = np.array(img.convert('L'))
        np_img = img / 127.5 - 1
        np_img = np.reshape(np_img, (1, im_heigth, im_width, 1))

        res = model.predict(np_img)
        label = res[0][0]
        crop = res[1][0]
        crop = (crop +1.) *127.5
        crop = crop.astype(np.uint8)[:, :, 0]

        # mask = res[2][0]
        # mask = mask * 255
        # mask = mask.astype(np.uint8)[:, :, 0]

        trueVin = savepath.split('/')[-1].split('.')[0]#readLabel(label)
        vin = readLabel(label)

        p = savepath.split('/')[:-1]
        p.append(trueVin + '.jpg')
        savepath = '/'.join(p)

        text_image = Image.new('L', (crop.shape[1], crop.shape[0] * 2))
        draw = ImageDraw.Draw(text_image)
        font = ImageFont.truetype('fonts/roboto/Roboto-Regular.ttf', 32)
        draw.text((0, 0), trueVin, font=font, fill=255)
        draw.text((0, crop.shape[0]+2), vin, font=font, fill=255)

        res = np.vstack((crop, np.array(text_image)))
        ratio = float(img.shape[1]/crop.shape[1])
        res = cv2.resize(res, (img.shape[1],int(res.shape[1]*ratio)))
        res = np.vstack((res, img))
        imsave(savepath, res)

    # sudo fuser -v /dev/nvidia*
