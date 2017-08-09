import numpy as np
from keras import callbacks
import random
import os
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from segmentmodel import getSegConvModel
# from model import bbox_model
# from scipy.misc import imsave
from utils import generate_timecode, generateVin
import cv2
im_height = 8 * 40
im_width = 16 * 40
input_shape = (im_height, im_width, 1)
nb_epoch = 50
y_coef = 0.1  # position of timecode from top or bottom
extra_pixels = 5

# random backgrounds
bcgs = []
bcgs_path = '/DATA/cocostuf/images/'

if True:
    bcgs = [os.path.join(root, f) for root, _, files in os.walk(bcgs_path) for f in files if f.endswith('.jpg')]

    with open('bcgs.txt', 'w') as f:
        for b in bcgs:
            f.write(b + '\n')
else:
    with open('bcgs.txt') as f:
        bcgs = f.readlines()
    bcgs = [b.strip() for b in bcgs]

fonts = [os.path.join(root, f) for root, _, files in os.walk('fonts/') for f in files if f.endswith('.ttf')]


def process(z):
    flips = [
        Image.FLIP_LEFT_RIGHT,
        Image.FLIP_TOP_BOTTOM,
        Image.ROTATE_90,
        Image.ROTATE_180,
        Image.ROTATE_270,
        Image.TRANSPOSE
    ]

    while True:
        tc = generate_timecode()
        ff = random.choice(fonts)

        try:
            font = ImageFont.truetype(ff, random.randint(16, 128))
        except:
            continue

        t_width, t_height = font.getsize(tc)

        if t_width < im_width and t_height < im_height * y_coef:
            try:
                bcg_img = Image.open(random.choice(bcgs)).convert('L')
            except:
                continue

            x = random.randint(0, im_width - t_width)
            if random.uniform(0, 1) > 0.5:
                y = random.randint(0, im_height * y_coef - t_height)
            else:
                y = random.randint(im_height * (1 - y_coef), im_height - t_height)

            bcg_img = bcg_img.rotate(random.uniform(0, 360))
            draw = ImageDraw.Draw(bcg_img)

            # random nuisance text
            for i in range(random.randint(0, 3)):
                txt = generateVin(random.randint(3, 10), 5)
                tx = random.randint(0, im_width)
                ty = random.randint(0, im_height)
                ImageDraw.Draw(bcg_img).text((tx, ty), txt, font=font, fill=random.randint(0, 255))

            # random background transform
            if random.uniform(0, 1) > 0.5:
                bcg_img = bcg_img.transpose(random.choice(flips))
            bcg_img = bcg_img.resize((im_width, im_height))

            color = random.randint(0, 255)

            # random black rect
            if random.uniform(0, 1) > 0.3:
                ImageDraw.Draw(bcg_img).rectangle(
                    [x - extra_pixels, y - extra_pixels, x + t_width + extra_pixels, y + t_height + extra_pixels],
                    fill=0)
                color = random.randint(128, 255)

            # time code text
            ImageDraw.Draw(bcg_img).text((x, y), tc, font=font, fill=color)

            sigma = random.uniform(0, 2)
            bcg_img = gaussian_filter(bcg_img, sigma)

            mask = Image.new('L', (im_width, im_height))
            draw = ImageDraw.Draw(mask)
            draw.text((x, y), tc, font=font, fill=255)
            mask = np.array(mask, dtype=np.float)
            mask[mask > 0] = 1

            bcg_img = np.array(bcg_img, dtype='uint8')

            # c = [float(x)/im_width, float(y)/im_height, float(x + t_width)/im_width, float(y + t_height)/im_height]

            # return [bcg_img, c]

            # imsave('tc_tmp/%s_%s_%s.jpg' % (tc, x, y), bcg_img)
            # cv2.imwrite('res/%s_%s_%s.jpg' % (tc, x, y), bcg_img)
            # cv2.imwrite('res/%s_%s_%s_mask.jpg' % (tc, x, y), mask*255)
            mask = np.packbits(np.asarray(mask, dtype='bool'), axis=-1)
            return [bcg_img, mask]


pool = ThreadPool(cpu_count() // 2)


def gen(batch_size=12):
    x = np.zeros((batch_size, im_height, im_width, 1), dtype='float32')
    y = np.zeros((batch_size, im_height, im_width, 1), dtype=np.float)
    # y = np.zeros((batch_size, 4))

    while True:
        result = pool.map(process, range(batch_size))
        for i, v in enumerate(result):
            x[i, :, :, 0] = v[0] / 127.5 - 1
            y[i, :, :, 0] = np.unpackbits(v[1], axis=-1)
            # y[i] = v[1]

        yield (x, y)


# model = bbox_model(input_shape, 4)
model = getSegConvModel(input_shape)
# model = make_parallel(model, 2)
model.compile('adam', 'binary_crossentropy')
# model.compile('adam', 'mse')

model.summary()
#model.load_weights('checkpoints/tc_segmenter_vl0.0022.hdf5')

print(len(fonts), len(bcgs))

model.fit_generator(generator=gen(),
                    validation_data=gen(),
                    steps_per_epoch=1000,
                    validation_steps=200,
                    epochs=nb_epoch,
                    max_q_size=100,
                    callbacks=[
                        callbacks.ModelCheckpoint(
                            'checkpoints/tc_segmenter_vl{val_loss:.4f}.hdf5',
                            monitor='val_loss')
                    ])
