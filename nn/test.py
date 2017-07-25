from sp_model import sp_model
from utils import vocabulary
from PIL import Image
import numpy as np
from scipy.misc import imsave
from utils import crop_rotate, mirror
import cv2
import os
import math
import itertools

im_heigth = 9*40
im_width = 16*40
input_shape = (im_heigth, im_width, 1)
max_angle = 45

model, transformerOutput, test_func = sp_model(input_shape, len(vocabulary)+2, returnTestFunctions=True)
model.load_weights('checkpoints/spatial_transformer_vl61.7531.hdf5')

imgs = [os.path.join(root, f) for root, _, files in os.walk('./tmp') for f in files if f.endswith('.jpg')]

def decode(predictions):
    out = predictions
    out_best = list(np.argmax(out[2:], 1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    outstr = ''
    for c in out_best:
        if c >= 0 and c < len(vocabulary):
            outstr += vocabulary[c]
    return outstr

for imgpath in imgs:
    img = Image.open(imgpath)
    img = img.resize((im_width, im_heigth))

    img = np.array(img.convert('L'))
    np_img = img / 127.5 - 1
    np_img = np.reshape(np_img, (1, im_heigth, im_width, 1))

    print(np_img.shape)

    result = test_func([np_img]) #model.predict(np_img)
    text = decode(result[0][0])
    print(text)
    transformed = ((transformerOutput([np_img])[0]+1.)*127.5).astype(np.uint8)[0,:,:,0]

    from scipy.misc import imsave
    imsave(imgpath.replace('tmp','res'), transformed)

    # sudo fuser -v /dev/nvidia*
