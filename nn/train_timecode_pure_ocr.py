import numpy as np
from keras import callbacks
import random
import os
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.filters import gaussian_filter
from timecode_ocr_model import timecode_ocr_model_small
from utils import generate_timecode
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from scipy.misc import imsave


vocabulary = '0123456789:;'

im_height = 32
im_width = 32*11
nb_epoch = 100
BATCH_SIZE = 8

tc_len = 11

y_coef = 0.1  # position of timecode from top or bottom
min_padding = -2
max_padding = 5
rand_transform = 5

# random backgrounds
bcgs = []
bcgs_path = '../../SUN_bcgs/'

if True:
    bcgs = [os.path.join(root, f) for root, _, files in os.walk(bcgs_path) for f in files if f.endswith('.jpg')]

    with open('bcgs.txt', 'w') as f:
        for b in bcgs:
            f.write(b + '\n')
else:
    with open('bcgs.txt') as f:
        bcgs = f.readlines()
    bcgs = [b.strip() for b in bcgs]

fonts = [os.path.join(root, f) for root, _, files in os.walk('monospaced/') for f in files if f.endswith('.ttf')]


def process(z):
    flips = [
        Image.FLIP_LEFT_RIGHT,
        Image.FLIP_TOP_BOTTOM,
        Image.ROTATE_90,
        Image.ROTATE_180,
        Image.ROTATE_270,
        Image.TRANSPOSE
    ]
    transforms = [
        [Image.EXTENT, 4],
        [Image.AFFINE, 6],
        [Image.PERSPECTIVE, 8],
        [Image.QUAD, 8]
    ]

    while True:
        tc = generate_timecode()
        ff = random.choice(fonts)

        try:
            font = ImageFont.truetype(ff, random.randint(16, 128))
            bcg_img = Image.open(random.choice(bcgs)).convert('L')
        except:
            continue

        t_width, t_height = font.getsize(tc)

        # final size
        txt_width = t_width + random.randint(min_padding, max_padding)
        txt_height = t_height + random.randint(min_padding, max_padding)

        # random bcg transforms
        if random.uniform(0, 1) > 0.5:
            bcg_img = bcg_img.transpose(random.choice(flips))

        # t = random.choice(transforms)
        # bcg_img = bcg_img.transform(
        #    (txt_width, txt_height), t[0], rand_transform * np.random.rand(t[1]))

        bcg_img = bcg_img.resize((txt_width, txt_height))

        color = random.randint(0, 255)

        x = random.randint(0, abs(txt_width - t_width))
        y = random.randint(0, abs(txt_height - t_height))

        # random black rect
        if random.uniform(0, 1) > 0.3:
            ImageDraw.Draw(bcg_img).rectangle([x, y, x + t_width, y + t_height], fill=0)
            color = random.randint(128, 255)

        # time code text
        ImageDraw.Draw(bcg_img).text((x, y), tc, font=font, fill=color)

        bcg_img = bcg_img.resize((im_width, im_height), Image.ANTIALIAS)

        sigma = random.uniform(0, 2)
        bcg_img = gaussian_filter(bcg_img, sigma)

        # random noise
        bcg_img = 0.1 * 255 * np.random.random(bcg_img.shape) + 0.9 * bcg_img

        bcg_img = np.array(bcg_img, dtype='uint8')

        # imsave('tc_tmp/%s_%s_%s.jpg' % (tc, x, y), bcg_img)
        # imsave('tmp.jpg', bcg_img)

        return [bcg_img, tc]


pool = ThreadPool(cpu_count() // 2)


def getLabel(s):
    label = np.zeros((tc_len, len(vocabulary)))
    for i, char in enumerate(s):
        label[i][vocabulary.index(char)] = 1
    return label


def gen(batch_size=1):
    x = np.zeros((batch_size, im_height, im_width, 1), dtype=np.float)
    y = np.zeros((batch_size, tc_len, len(vocabulary)))

    while True:
        result = pool.map(process, range(batch_size))
        for i, v in enumerate(result):
            x[i, :, :, 0] = v[0] / 127.5 - 1
            y[i] = getLabel(v[1])

        yield (x, y)

model = timecode_ocr_model_small()
#model.load_weights('checkpoints/OCRmodel_vl0.1468.hdf5')
# model.summary()

model.fit_generator(generator=gen(batch_size=BATCH_SIZE),
                    validation_data=gen(batch_size=BATCH_SIZE),
                    steps_per_epoch=3000,
                    validation_steps=100,
                    epochs=nb_epoch,
                    max_q_size=100,
                    callbacks=[
                        callbacks.ModelCheckpoint(
                            'checkpoints/OCRmodel_vl{val_loss:.4f}.hdf5',
                            monitor='val_loss')
                    ])
