from model import bbox_model
from PIL import Image
import numpy as np
from scipy.misc import imsave

im_heigth = 9*40
im_width = 16*40
input_shape = (im_heigth, im_width, 1)

model = bbox_model(shape=input_shape, coords_count=5)
model.load_weights('checkpoints/vl0.0045.hdf5')

img = Image.open('../20170703_165336.jpg')
img = img.resize((im_width, im_heigth))

img = img.convert('L')
img = np.array(img, dtype='float32') / 127.5 - 1
img = np.reshape(img, (1, im_heigth, im_width, 1))

print(img.shape)

result = model.predict(img)
r = result[0]
print(r)
print(int(r[1] * im_heigth), int(r[3] * im_heigth), int(r[0] * im_width), int(r[2] * im_width))
crop = img[
       0,
       max(int(r[1] * im_heigth), 0): max(int(r[3] * im_heigth), 0),
       max(int(r[0] * im_width), 0): max(int(r[2] * im_width), 0),
       0]

print(crop.shape)

imsave('tmp/crop.jpg', crop)

# sudo fuser -v /dev/nvidia*
