import numpy as np
from keras import callbacks
import random
import os
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from bboxModel import bboxModel
from att_ocrModel import getOCRModel
# from model import bbox_model
# from scipy.misc import imsave
from utils import generate_timecode, generateVin
import cv2
im_height = 8 * 40
im_width = 16 * 40
input_shape = (im_height, im_width, 1)

mask_width = 80
mask_height = 40

nb_epoch = 50
y_coef = 0.3  # position of timecode from top or bottom
extra_pixels = 5

# random backgrounds
bcgs = []
bcgs_path = '/DATA/MASK/data/coco/train2014/'

# bboxExtractor = bboxModel(test=True)
# bboxExtractor.load_weights('checkpoints/BBoxModel_vl0.2721.hdf5')
# bboxExtractor._make_predict_function() # yobana zahachka

vocabulary = '0123456789:;'

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


def drawBBox(bbox, img):
    tl_x = int((bbox[0]-(bbox[2]/2))*img.shape[1])
    tl_y = int((bbox[1]-(bbox[3]/2))*img.shape[0])
    br_x = int((bbox[0]+(bbox[2]/2))*img.shape[1])
    br_y = int((bbox[1]+(bbox[3]/2))*img.shape[0])

    cv2.rectangle(img, (tl_x, tl_y), (br_x, br_y), (255,0,0), thickness=1, lineType=8, shift=0)
    return img

def cropBBox( img,bbox):
    tl_x = max(int((bbox[0]-(bbox[2]/2))*img.shape[1]),0)
    tl_y = max(int((bbox[1]-(bbox[3]/2))*img.shape[0]),0)
    br_x = int((bbox[0]+(bbox[2]/2))*img.shape[1])
    br_y = int((bbox[1]+(bbox[3]/2))*img.shape[0])

    crop = img[tl_y:br_y,tl_x:br_x]
    crop = cv2.resize(crop,(16*32,32), interpolation=cv2.INTER_CUBIC)
    return crop

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
            font = ImageFont.truetype(ff, random.randint(8, 16))
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

            color = random.randint(90, 255)

            # random black rect
            if random.uniform(0, 1) > 0.3:
                ImageDraw.Draw(bcg_img).rectangle(
                    [x - extra_pixels, y - extra_pixels, x + t_width + extra_pixels, y + t_height + extra_pixels],
                    fill=0)
                color = random.randint(128, 255)

            if random.uniform(0, 1) < 0.3:
                ImageDraw.Draw(bcg_img).rectangle(
                    [0, 0, im_width, im_height*random.uniform(0.01, 0.2)], fill=0)
                ImageDraw.Draw(bcg_img).rectangle(
                    [0, im_height * random.uniform(0.8, 0.99), im_width, im_height], fill=0)

            # time code text
            ImageDraw.Draw(bcg_img).text((x, y), tc, font=font, fill=color)

            sigma = random.uniform(0, 0.3)
            bcg_img = gaussian_filter(bcg_img, sigma)


            bcg_img = np.array(bcg_img, dtype='uint8')

            cv2.imwrite('tmp/%s_%s_%s_timecode.jpg' % (tc, x, y), bcg_img)

            bbox = [float(x * random.uniform(0.996,1.01) + t_width*1.1/ 2) / im_width,
                    float(y * random.uniform(0.998,1.02) + t_height * 1.15 / 2) / im_height,
                    float(t_width * 1.20) / im_width,
                    float(t_height * 1.20 ) / im_height]


            crop = cropBBox(bcg_img, bbox)
            #cv2.imwrite('res/%s_%s_%s_crop.jpg' % (tc, x, y), crop)
            return [crop, tc]


pool = ThreadPool(max(cpu_count() // 2, 4))


def getLabel(s):
    label = np.zeros((11, len(vocabulary)))
    for i, char in enumerate(s):
        label[i][vocabulary.index(char)] = 1
    return label

def gen(batch_size=8):
    x = np.zeros((batch_size, 32, 512, 1), dtype='float32')
    y = np.zeros((batch_size, 11,len(vocabulary)), dtype=np.float)

    while True:
        result = pool.map(process, range(batch_size))
        for i, v in enumerate(result):
            x[i, :, :, 0] = v[0] / 127.5 - 1
            y[i] = getLabel(v[1])

        yield (x, y)


model = getOCRModel()
model.summary()
#model.load_weights('checkpoints/tc_segmenter_vl0.7355.hdf5')

print(len(fonts), len(bcgs))

model.fit_generator(generator=gen(),
                    validation_data=gen(),
                    steps_per_epoch=1000,
                    validation_steps=200,
                    epochs=nb_epoch,
                    max_q_size=100,
                    callbacks=[
                        callbacks.ModelCheckpoint(
                            'checkpoints/BBoxModel_vl{val_loss:.4f}.hdf5',
                            monitor='val_loss')
                    ])
