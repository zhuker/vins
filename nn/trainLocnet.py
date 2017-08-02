import numpy as np
from keras import callbacks
import random
import os
from PIL import Image, ImageFont, ImageDraw, ImageOps
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import Pool, cpu_count
from sp_model import sp_model, locnet_on_mask, getlocnet, testMatrixModel
from utils import generateVin, rotate_coord, getLabel, vocabulary
import cv2

im_heigth = 9 * 40
im_width = 16 * 40
input_shape = (im_heigth, im_width, 1)
coords_count = 5
nb_epoch = 500
vin_length = 17
max_angle = 45
max_spaces = 0
maxLength = vin_length + max_spaces

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



def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def getMatrix(pts):
    rect = pts.astype(np.float32)

    origHeight = 9 * 40
    origWidth = 16 * 40

    dst = np.array([
        [-1, -1],
        [-1, 1],
        [1, 1],
        [1, -1]], dtype="float32")

    rect[:, 0] /= origWidth/2
    rect[:, 1] /= origHeight/2

    rect[:, 0] -= 1
    rect[:, 1] -= 1

    M = cv2.getAffineTransform( dst[:3], rect[:3])
    return M



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

    def getLabel(vin):
        label = np.zeros((17, len(vocabulary)))
        for i, char in enumerate(vin):
            label[i][vocabulary.index(char)] = 1
        return label

    while True:
        vin = generateVin(vin_length, max_spaces)
        if len(vin) < maxLength:
            for i in range(maxLength - len(vin)):
                vin += ' '
        ff = random.choice(fonts)

        font = ImageFont.truetype(ff, random.randint(32, 256))
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

        rot_coords[0] = rotate_coord(0, (t_img_size - t_height) / 2., t_img_size / 2., t_img_size / 2., rot_deg)
        rot_coords[1] = rotate_coord(0, (t_img_size + t_height) / 2., t_img_size / 2., t_img_size / 2., rot_deg)
        rot_coords[2] = rotate_coord(t_img_size, (t_img_size + t_height) / 2., t_img_size / 2., t_img_size / 2., rot_deg)
        rot_coords[3] = rotate_coord(t_img_size, (t_img_size - t_height) / 2., t_img_size / 2., t_img_size / 2., rot_deg)

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

            x = random.randint(-min_tx, im_width - max_tx)  # int((-min_tx + im_width-max_tx)/2)
            y = random.randint(-min_ty, im_heigth - max_ty)  # int((-min_ty + im_heigth-max_ty)/2)

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

            mask = Image.new('L', (32*16, 32))
            draw = ImageDraw.Draw(mask)
            font.size =32
            draw.text((0,0), vin, font=font, fill=255)
            mask = np.array(mask, dtype=np.float)
            mask[mask > -1] = 1
            mask = np.expand_dims(mask, axis=-1)




            # output endpoints and height to define rectangle
            c = [
                2 * (rot_coords[0, 0] / im_width) - 1,
                2 * (rot_coords[0, 1] / im_heigth) - 1,
                2 * (rot_coords[2, 0] / im_width) - 1,
                2 * (rot_coords[2, 1] / im_heigth) - 1,
                # m_point_a[0]/im_width,
                # m_point_a[1]/im_heigth,
                # m_point_b[0]/im_width,
                # m_point_b[1]/im_heigth,
                rot_deg / max_angle
                # t_height#/t_width
            ]

            # crop = crop_rotate(bcg_img, -rot_deg, rot_coords[0],rot_coords[2])
            #
            # from scipy.misc import imsave
            # try:
            #     imsave('tmp/%s_%s_%s.jpg' % (vin, x, y), crop)
            # except:
            #     print('trololo')

            label = getLabel(vin)
            matrix = getMatrix(rot_coords)

            # crop = cv2.warpAffine(bcg_img, matrix, (1000, 1000))
            # cv2.imwrite('temp_crop.png',crop)
            return np.reshape(bcg_img, (im_heigth, im_width, 1)), label, mask, matrix.flatten()


pool = Pool(cpu_count() // 2)

def gen(batch_size=4):
    x = np.zeros((batch_size, im_heigth, im_width, 1), dtype='float32')
    y = np.zeros((batch_size, 17, len(vocabulary)))
    masks = np.zeros((batch_size, 32, 32*16, 1))
    matrix = np.zeros((batch_size, 6))
    while True:
        result = pool.map(process, range(batch_size))
        for i, v in enumerate(result):
            x[i] = v[0] / 127.5 - 1
            y[i] = v[1]
            masks[i] = v[2]
            matrix[i] = v[3]
        # crop = testMatrixModel.predict(x)[0]
        # crop += 1
        # crop *= 127.5
        # crop = crop[:,:,0].astype(np.uint8)
        # img = cv2.resize(((x[0]+1)*127.5)[:,:,0].astype(np.uint8),(crop.shape[1], int(360*crop.shape[1]/float(640))))
        #
        # cv2.imwrite('res/'+str(z)+'.jpg', np.vstack((crop,img)))
        # z+=1
        yield (x, matrix)



model =  getlocnet(input_shape)#, locnet_on_mask(input_shape)
#model.compile('adam', 'mse')
#model.load_weights('checkpoints/endToEnd_vl0.0747.hdf5')

# model = bbox_model(shape=input_shape, coords_count=coords_count)
# model = to_multi_gpu(model, 2)

# model.compile(loss="mean_squared_error", optimizer='sgd')
model.summary()

model.fit_generator(generator=gen(),
                    validation_data=gen(),
                    steps_per_epoch=3000,
                    validation_steps=100,
                    epochs=nb_epoch,
                    max_q_size=100,
                    callbacks=[
                        callbacks.ModelCheckpoint(
                            'checkpoints/LOCNET_vl{val_loss:.4f}.hdf5',
                            monitor='val_loss')
                    ])
