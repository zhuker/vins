import numpy as np
from keras import callbacks
import random
import os
import cv2
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.filters import gaussian_filter
from timecode_ocr_model import timecode_ocr_model
from segmentmodel import getSegModel
from utils import generate_timecode, generateVin
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from adjustTextPosition import adjust_horizontal, adjust
from scipy.misc import imsave


vocabulary = '0123456789:;'

im_height = 32
im_width = 32*16
nb_epoch = 100
BATCH_SIZE = 8

tc_len = 11

input_height = 8*40
input_width = 16*40
input_shape = (input_height, input_width, 1)
y_coef = 0.1  # position of timecode from top or bottom
extra_pixels = 5

seg_model = getSegModel(input_shape)
seg_model.compile('adam', 'binary_crossentropy')
seg_model.load_weights('checkpoints/tc_segmenter_vl0.0022.hdf5')

# yobana zahachka
seg_model._make_predict_function()

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

    while True:
        tc = generate_timecode()
        ff = random.choice(fonts)

        try:
            font = ImageFont.truetype(ff, random.randint(16, 128))
        except:
            continue

        t_width, t_height = font.getsize(tc)

        if t_width < input_width and t_height < input_height * y_coef:
            try:
                bcg_img = Image.open(random.choice(bcgs)).convert('L')
            except:
                continue

            x = random.randint(0, input_width - t_width)
            if random.uniform(0, 1) > 0.5:
                y = random.randint(0, input_height * y_coef - t_height)
            else:
                y = random.randint(input_height * (1 - y_coef), input_height - t_height)

            bcg_img = bcg_img.rotate(random.uniform(0, 360))

            # random nuisance text
            # for i in range(random.randint(0, 3)):
            #     txt = generateVin(random.randint(3, 10), 5)
            #     tx = random.randint(0, input_width)
            #     ty = random.randint(0, input_height)
            #     ImageDraw.Draw(bcg_img).text((tx, ty), txt, font=font, fill=random.randint(0, 255))

            # random background transform
            if random.uniform(0, 1) > 0.5:
                bcg_img = bcg_img.transpose(random.choice(flips))
            bcg_img = bcg_img.resize((input_width, input_height))

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

            bcg_img = np.array(bcg_img, dtype='uint8')

            # imsave('tc_tmp/%s_%s_%s.jpg' % (tc, x, y), bcg_img)

            return [bcg_img, tc]


pool = ThreadPool(cpu_count() // 2)


def getLabel(s):
    label = np.zeros((tc_len, len(vocabulary)))
    for i, char in enumerate(s):
        label[i][vocabulary.index(char)] = 1
    return label


def gen(batch_size=1):
    input_imgs = np.zeros((batch_size, input_height, input_width, 1), dtype=np.float32)
    orig_imgs = np.zeros((batch_size, input_height, input_width, 1), dtype=np.uint8)
    x = np.zeros((batch_size, im_height, im_width, 1), dtype=np.float)
    y = np.zeros((batch_size, tc_len, len(vocabulary)))

    while True:
        result = pool.map(process, range(batch_size))
        for i, v in enumerate(result):
            input_imgs[i, :, :, 0] = v[0] / 127.5 - 1
            orig_imgs[i, :, :, 0] = v[0]

        pred_masks = seg_model.predict(input_imgs)
        for i, mask in enumerate(pred_masks):
            mask = np.array(mask * 255, dtype=np.uint8)
            # imsave('tmp_mask.jpg', mask[:, :, 0])
            # print(mask.shape)
            crop, crop_mask = adjust_horizontal(orig_imgs[i], mask)
            # print(crop.shape)
            crop = cv2.resize(crop, (im_width, im_height))
            x[i, :, :, 0] = crop / 127.5 - 1
            y[i] = getLabel(result[i][1])

        yield (x, y)

model = timecode_ocr_model()
model.load_weights('checkpoints/OCRmodel_vl0.3992.hdf5')
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
