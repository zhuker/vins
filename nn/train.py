import numpy as np
from keras import callbacks
import random
import os
from PIL import Image, ImageFont, ImageDraw, ImageOps
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import Pool, cpu_count
from model import bbox_model
from utils import rotate_coord, crop_rotate

im_heigth = 9*40
im_width = 16*40
input_shape = (im_heigth, im_width, 1)
coords_count = 5
nb_epoch = 50
vin_length = 17

# random backgrounds
bcgs = []
bcgs_path = '/DATA/cars/'


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

    alphabet = '0123456789ABCDEFGHJKLMNPRSTUVWXYZ'
    rot_coords = np.zeros((4, 2), dtype='float32')
    flips = [
        Image.FLIP_LEFT_RIGHT,
        Image.FLIP_TOP_BOTTOM,
        Image.ROTATE_90,
        Image.ROTATE_180,
        Image.ROTATE_270,
        Image.TRANSPOSE
    ]

    while True:
        vin = ''.join(random.choice(alphabet) for _ in range(vin_length))

        ff = random.choice(fonts)

        font = ImageFont.truetype(ff, random.randint(32, 256))
        t_width, t_height = font.getsize(vin)

        # text image is a square
        t_img_size = max(t_height, t_width)

        # rot_deg = random.uniform(0, 360)
        rot_deg = random.uniform(-10, 10)

        # text image will have original text located horizontally in the middle
        # original coordinates will be (0, (s-h)/2) - top left and (s, (s+h)/2) - bottom right
        # here s denotes text image size (image width), h - text height
        # we rotate all four corners of text around the center of the image (s/2, s/2) on the chosen angle
        # and check if rotated text can be fitted in background image

        rot_coords[0] = rotate_coord(0, (t_img_size - t_height)/2., t_img_size/2., t_img_size/2., rot_deg)
        rot_coords[1] = rotate_coord(0, (t_img_size + t_height)/2., t_img_size/2., t_img_size/2., rot_deg)
        rot_coords[2] = rotate_coord(t_img_size, (t_img_size + t_height)/2., t_img_size/2., t_img_size/2., rot_deg)
        rot_coords[3] = rotate_coord(t_img_size, (t_img_size - t_height)/2., t_img_size/2., t_img_size/2., rot_deg)

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
            draw.text((0, round(t_img_size/2. - t_height/2.)), vin, font=font, fill=255)
            text_image = text_image.rotate(rot_deg)

            x = random.randint(-min_tx, im_width - max_tx)
            y = random.randint(-min_ty, im_heigth - max_ty)

            color = random.randint(0, 255)

            if random.uniform(0, 1) > 0.5:
                bcg_img = bcg_img.transpose(random.choice(flips))
            bcg_img = bcg_img.resize((im_width, im_heigth))
            colorization = ImageOps.colorize(text_image, (0, 0, 0), (color, color, color))
            bcg_img.paste(colorization, (x, y), text_image)

            sigma = random.uniform(0, 2)
            bcg_img = gaussian_filter(bcg_img, sigma)

            # from scipy.misc import imsave
            # imsave('tmp/%s_%s_%s.jpg' % (vin, x, y), bcg_img)

            bcg_img = np.array(bcg_img, dtype='uint8')

            rot_coords[:] += [x, y]

            # bcg_img = cv2.polylines(bcg_img, np.int32([rot_coords]), 1, (255,255,255))

            # end points of center line going through text
            # m_point_a = (rot_coords[0] + rot_coords[1])/2
            # m_point_b = (rot_coords[2] + rot_coords[3])/2

            # output endpoints and height to define rectangle
            c = [
                rot_coords[0, 0]/im_width,
                rot_coords[0, 1]/im_heigth,
                rot_coords[2, 0]/im_width,
                rot_coords[2, 1]/im_heigth,
                # m_point_a[0]/im_width,
                # m_point_a[1]/im_heigth,
                # m_point_b[0]/im_width,
                # m_point_b[1]/im_heigth,
                rot_deg / 360.
                #t_height#/t_width
            ]

            # crop = crop_rotate(bcg_img, -rot_deg, rot_coords[0],rot_coords[2])
            #
            # from scipy.misc import imsave
            # try:
            #     imsave('tmp/%s_%s_%s.jpg' % (vin, x, y), crop)
            # except:
            #     print('trololo')

            return [np.reshape(bcg_img, (im_heigth, im_width, 1)), c]


pool = Pool(cpu_count() // 2)


def gen(batch_size=8):
    x = np.zeros((batch_size, im_heigth, im_width, 1), dtype='float32')
    y = np.zeros((batch_size, coords_count), dtype='float32')
    while True:
        result = pool.map(process, range(batch_size))
        #for i in range(batch_size):
        #    v = process(i)
        for i, v in enumerate(result):
            x[i] = v[0] / 127.5 - 1
            y[i] = v[1]

        yield (x, y)

model = bbox_model(shape=input_shape, coords_count=coords_count)
# model = to_multi_gpu(model, 2)

model.compile(loss="mean_squared_error", optimizer='adam')
model.load_weights('checkpoints/vl0.0101.hdf5')
model.summary()

model.fit_generator(generator=gen(),
                    validation_data=gen(),
                    steps_per_epoch=3000,
                    validation_steps=100,
                    epochs=nb_epoch,
                    max_q_size=100,
                    callbacks=[
                        callbacks.ModelCheckpoint(
                            'checkpoints/vl{val_loss:.4f}.hdf5',
                            monitor='val_loss')
                    ])
