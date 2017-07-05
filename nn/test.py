from model import bbox_model
from PIL import Image
import numpy as np
from scipy.misc import imsave

im_heigth = 9*50
im_width = 16*50
input_shape = (im_heigth, im_width, 1)

model = bbox_model(shape=input_shape)
model.load_weights('checkpoints/vl0.0179.hdf5')

img = Image.open('../20170703_165336.jpg')
img = img.resize((im_width, im_heigth))

img = img.convert('L')
img = np.array(img, dtype='float32') / 127.5 - 1
img = np.reshape(img, (1, im_heigth, im_width, 1))

result = model.predict(img)
r = result[0]
print(r)

crop = img[0, int(r[1] * im_heigth): int(r[3] * im_heigth), int(r[0] * im_width): int(r[2] * im_width), 0]

imsave('tmp/crop.jpg', crop)

# sudo fuser -v /dev/nvidia*
