from model import bbox_model
from PIL import Image
import numpy as np
from scipy.misc import imsave
from utils import crop_rotate, mirror
import cv2
import os
import math

im_heigth = 9*40
im_width = 16*40
input_shape = (im_heigth, im_width, 1)
max_angle = 45

model = bbox_model(shape=input_shape, coords_count=5)
model.load_weights('checkpoints/vl0.0385.hdf5')

imgs = [os.path.join(root, f) for root, _, files in os.walk('./tmp') for f in files if f.endswith('.jpg')]
for imgpath in imgs:
    img = Image.open(imgpath)
    img = img.resize((im_width, im_heigth))

    img = np.array(img.convert('L'))
    np_img = img / 127.5 - 1
    np_img = np.reshape(np_img, (1, im_heigth, im_width, 1))

    print(np_img.shape)

    result = model.predict(np_img)
    r = result[0]
    r[:4] += 1
    r[:4] /= 2
    print(r)
    print(int(r[1] * im_heigth), int(r[3] * im_heigth), int(r[0] * im_width), int(r[2] * im_width))
    #crop = crop_rotate(img, -r[4]*max_angle, (r[0]*im_width, r[1]*im_heigth), (r[2]*im_width, r[3]*im_heigth))

    angle = max_angle*r[4]
    a = [r[0]*im_width, r[1]*im_heigth]
    c = [r[2] * im_width, r[3] * im_heigth]
    b, d = mirror(a,c, angle)
    img = cv2.polylines(np.array(img, dtype=np.uint8),
                        np.array([[a,
                                   b,
                                   c,
                                   d]], dtype=np.int32), 1, (255,255,255))
    from scipy.misc import imsave
    imsave(imgpath.replace('tmp','res'), img)

    # sudo fuser -v /dev/nvidia*
