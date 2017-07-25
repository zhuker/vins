import numpy as np
from keras import callbacks
import random
import os
from PIL import Image, ImageFont, ImageDraw, ImageOps
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from segmentmodel import getSegModel
from utils import generateVin, rotate_coord
from multi_gpu import make_parallel

im_heigth = 9 * 40
im_width = 16 * 40
input_shape = (im_heigth, im_width, 1)
coords_count = 5
nb_epoch = 50
vin_length = 17
max_angle = 45.
max_spaces = 0
maxLength = vin_length + max_spaces

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

fonts = [os.path.join(root, f) for root, _, files in os.walk('fonts/') for f in files if f.endswith('.ttf')]


def process(z):
    rot_coords = np.zeros((4, 2), dtype='float32')
    flips = [
        Image.FLIP_LEFT_RIGHT,
        Image.FLIP_TOP_BOTTOM,
        Image.ROTATE_90,
        Image.ROTATE_180,
        Image.ROTATE_270,
        Image.TRANSPOSE
    ]
    k = 0
    while True:
        k += 1
        vin = generateVin(vin_length, max_spaces)
        if len(vin) < maxLength:
            for i in range(maxLength - len(vin)):
                vin += ' '
        ff = random.choice(fonts)

        font = ImageFont.truetype(ff, random.randint(16, 128))
        t_width, t_height = font.getsize(vin)

        # text image is a square
        t_img_size = max(t_height, t_width)

        # rot_deg = random.uniform(0, 360)
        rot_deg = random.uniform(-max_angle, max_angle)

        # text image will have original text located horizontally in the middle
        # original coordinates will be (0, (s-h)/2) - top left and (s, (s+h)/2) - bottom right
        # here s denotes text image size (image width), h - text height
        # we rotate all four corners of text around the center of the image (s/2, s/2) on the chosen angle
        # and check if rotated text can be fitted in background image

        rot_coords[0] = rotate_coord(
            0, (t_img_size - t_height) / 2., t_img_size / 2., t_img_size / 2., rot_deg)
        rot_coords[1] = rotate_coord(
            0, (t_img_size + t_height) / 2., t_img_size / 2., t_img_size / 2., rot_deg)
        rot_coords[2] = rotate_coord(
            t_img_size, (t_img_size + t_height) / 2., t_img_size / 2., t_img_size / 2., rot_deg)
        rot_coords[3] = rotate_coord(
            t_img_size, (t_img_size - t_height) / 2., t_img_size / 2., t_img_size / 2., rot_deg)

        min_tx = round(np.min(rot_coords[:, 0]))
        max_tx = round(np.max(rot_coords[:, 0]))
        min_ty = round(np.min(rot_coords[:, 1]))
        max_ty = round(np.max(rot_coords[:, 1]))

        dx = max_tx - min_tx
        dy = max_ty - min_ty

        if dx < im_width and dy < im_heigth:
            try:
                bcg_img = Image.open(random.choice(bcgs)).convert('L')
            except:
                continue

            text_image = Image.new('L', (t_img_size, t_img_size))
            draw = ImageDraw.Draw(text_image)
            draw.text((0, round(t_img_size / 2. - t_height / 2.)), vin, font=font, fill=255)
            text_image = text_image.rotate(rot_deg)

            x = random.randint(-min_tx, im_width - max_tx)
            y = random.randint(-min_ty, im_heigth - max_ty)

            color = random.randint(2, 255)

            if random.uniform(0, 1) > 0.5:
                bcg_img = bcg_img.transpose(random.choice(flips))
            bcg_img = bcg_img.resize((im_width, im_heigth))
            colorization = ImageOps.colorize(text_image, (0, 0, 0), (color, color, color))
            bcg_img.paste(colorization, (x, y), text_image)

            sigma = random.uniform(0, 2)
            bcg_img = gaussian_filter(bcg_img, sigma)

            mask = Image.new('L', (im_width, im_heigth))
            colorization = ImageOps.colorize(text_image, (0, 0, 0), (255, 255, 255))
            mask.paste(colorization, (x, y), text_image)
            mask = np.array(mask, dtype=np.float)
            mask[mask > 0] = 1

            bcg_img = np.array(bcg_img, dtype='uint8')

            # from scipy.misc import imsave
            # imsave('tmp/%s_%s_%s.jpg' % (vin, x, y), bcg_img)

            # mask = mask.reshape((im_heigth, im_width, 1))

            rot_coords[:] += [x, y]
            # print k
            # return [np.reshape(bcg_img, (im_heigth, im_width, 1)), mask]
            mask = np.packbits(np.asarray(mask, dtype='bool'), axis=-1)
            return [bcg_img, mask]


pool = ThreadPool(8)


def gen(batch_size=16):
    x = np.zeros((batch_size, im_heigth, im_width, 1), dtype='float32')
    y = np.zeros((batch_size, im_heigth, im_width, 1), dtype=np.float)

    while True:
        result = pool.map(process, range(batch_size))
        for i, v in enumerate(result):
            x[i, :, :, 0] = v[0] / 127.5 - 1
            y[i, :, :, 0] = np.unpackbits(v[1], axis=-1)

        yield (x, y)


model = getSegModel(input_shape)
# model = make_parallel(model, 2)
model.compile('adam', 'binary_crossentropy')

# model.load_weights('checkpoints/spatial_transformer_vl0.5295.hdf5')
model.summary()
# model.load_weights('checkpoints/segmenter_vl0.0228.hdf5')

model.fit_generator(generator=gen(),
                    validation_data=gen(),
                    steps_per_epoch=500,
                    validation_steps=100,
                    epochs=nb_epoch,
                    max_q_size=100,
                    callbacks=[
                        callbacks.ModelCheckpoint(
                            'checkpoints/segmenter_vl{val_loss:.4f}.hdf5',
                            monitor='val_loss')
                    ])
