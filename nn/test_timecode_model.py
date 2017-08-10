from segmentmodel import getSegConvModelNopool
import numpy as np
import cv2
import os
from scipy.misc import imsave
from adjustTextPosition import adjust, adjust_horizontal
from timecode_ocr_model import timecode_ocr_model_small
from PIL import Image, ImageFont, ImageDraw


im_height = 8*40
im_width = 16*40
input_shape = (im_height, im_width, 1)
max_angle = 45

seg_model = getSegConvModelNopool(input_shape)
seg_model.compile('adam', 'binary_crossentropy')
seg_model.load_weights('checkpoints/tc_segmenter_vl0.0024.hdf5')

ocr_model = timecode_ocr_model_small()
ocr_model.load_weights('checkpoints/OCRmodel_vl0.0424.hdf5')

imgs = [os.path.join(root, f) for root, _, files in os.walk('sdi/R_5209B6_ENG_EM') for f in files if f.lower().endswith('.jpg')]
imgs.sort()

vocabulary = '0123456789:;'


def readLabel(label):
    vin = ''
    for charprobs in label:
        charid = np.argmax(charprobs)
        vin += vocabulary[charid]
    return vin

for imgpath in imgs:
    savepath = imgpath.replace('R_5209B6_ENG_EM', 'R_5209B6_ENG_EM_res')
    img = Image.open(imgpath)
    img = img.resize((im_width, im_height))

    img = np.array(img.convert('L')).reshape((im_height, im_width, 1))
    np_img = img / 127.5 - 1
    np_img = np.reshape(np_img, (1, im_height, im_width, 1))

    mask = seg_model.predict(np_img)[0]
    mask *= 255
    mask = mask.astype(np.uint8)[:, :, 0]

    # mask = np.vstack((img, mask))

    res, newmask = adjust_horizontal(img, mask)

    imsave('tmp.jpg', res[:, :, 0])

    cropVin = np.expand_dims(np.array([cv2.resize(res, (32 * 11, 32)).astype(np.float) / 127.5 - 1]), axis=-1)
    label = ocr_model.predict(cropVin)[0]
    tc = readLabel(label)

    # text_image = Image.new('L', (cropVin.shape[2], cropVin.shape[1] * 2))
    # draw = ImageDraw.Draw(text_image)
    # font = ImageFont.truetype('fonts/roboto/Roboto-Regular.ttf', 32)
    # draw.text((0, 0), tc, font=font, fill=255)
    #
    # res = np.vstack((cv2.resize(res, (cropVin.shape[2], cropVin.shape[1])), np.array(text_image)))

    text_img = Image.new('L', (im_width, 32))
    draw = ImageDraw.Draw(text_img)
    font = ImageFont.truetype('fonts/roboto/Roboto-Regular.ttf', 32)
    draw.text((0, 0), tc, font=font, fill=255)

    mask_img = np.zeros((32, im_width))
    mask_img[:, 0:352] = 127.5 * (cropVin[0, :, :, 0] + 1)

    res = np.vstack((img[:, :, 0], mask, mask_img, text_img))

    imsave(savepath, res)
    print(savepath)

    # sudo fuser -v /dev/nvidia*
