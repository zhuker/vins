from segmentmodel import getSegModel
from PIL import Image
import numpy as np
import os
from multi_gpu import make_parallel

im_heigth = 9*40
im_width = 16*40
input_shape = (im_heigth, im_width, 1)
max_angle = 45

model = getSegModel(input_shape)
# model = make_parallel(model, 1)
model.compile('adam', 'binary_crossentropy')
model.load_weights('checkpoints/segmenter_vl0.0376.hdf5', by_name=True)

imgs = [os.path.join(root, f) for root, _, files in os.walk('./tmp') for f in files if f.lower().endswith('.jpg')]


for imgpath in imgs:
    img = Image.open(imgpath)
    img = img.resize((im_width, im_heigth))

    img = np.array(img.convert('L'))
    np_img = img / 127.5 - 1
    np_img = np.reshape(np_img, (1, im_heigth, im_width, 1))

    mask = model.predict(np_img)[0]
    mask *= 255

    from scipy.misc import imsave
    imsave(imgpath.replace('tmp', 'res'), mask.astype(np.uint8)[:, :, 0])

    # sudo fuser -v /dev/nvidia*
