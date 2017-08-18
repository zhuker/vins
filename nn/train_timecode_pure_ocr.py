import numpy as np
from keras import callbacks
import random
import os
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.filters import gaussian_filter
from timecode_ocr_model import timecode_ocr_model_RGB_full
from utils import generate_timecode
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from scipy.misc import imsave
import cv2
from utils import generate_timecode
vocabulary = '0123456789:;'
randomVocab = ' 1234567890ABCDEFGHJKLMNPRSTUVWXYZabcderfgklmnopqrstuv           '

im_height = 8 * 40
im_width = 16 * 40
nb_epoch = 100
BATCH_SIZE = 16
extra_pixels = 5
tc_len = 11

y_coef = 0.1  # position of timecode from top or bottom
min_padding = -2
max_padding = 5
rand_transform = 5

# random backgrounds
bcgs = []

bcgs_path = '../../bcgs/'
bcgs_path = '/DATA/MASK/data/coco/train2014/'

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

def cropBBox( img,bbox):
    tl_x = max(int((bbox[0]-(bbox[2]/2))*img.shape[1]),0)
    tl_y = max(int((bbox[1]-(bbox[3]/2))*img.shape[0]),0)
    br_x = int((bbox[0]+(bbox[2]/2))*img.shape[1])
    br_y = int((bbox[1]+(bbox[3]/2))*img.shape[0])

    crop = img[tl_y:br_y,tl_x:br_x]
    crop = cv2.resize(crop,(27*7,27), interpolation=cv2.INTER_CUBIC)
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
            font = ImageFont.truetype(ff, random.randint(9, 18))
        except:
            continue

        t_width, t_height = font.getsize(tc)

        if t_width < im_width and t_height < im_height * y_coef:
            try:
                bcg_img = Image.open(random.choice(bcgs)).convert('RGB')
            except:
                continue

            x = random.randint(0, im_width - t_width)
            if random.uniform(0, 1) > 0.5:
                y = random.randint(0, im_height * y_coef - t_height)
            else:
                y = random.randint(im_height * (1 - y_coef), im_height - t_height)

            bcg_img = bcg_img.rotate(random.uniform(0, 360))

            # random nuisance text
            for i in range(random.randint(0, 15)):
                txt = ''.join([random.choice(randomVocab) for i in range(0,random.randint(4,19))])
                tx = random.randint(0, im_width)
                ty = random.randint(0, im_height)
                ImageDraw.Draw(bcg_img).text((tx, ty), txt, font=font, fill=(random.randint(180, 255),random.randint(180, 255),random.randint(180, 255)))

            # random background transform
            if random.uniform(0, 1) > 0.5:
                bcg_img = bcg_img.transpose(random.choice(flips))
            bcg_img = bcg_img.resize((im_width, im_height))

            color = random.randint(150, 255)

            # random black rect
            if random.uniform(0, 1) > 0.6:
                ImageDraw.Draw(bcg_img).rectangle(
                    [x - extra_pixels, y - extra_pixels, x + t_width + extra_pixels, y + t_height + extra_pixels],
                    fill=0)
                color = random.randint(150, 255)

            if random.uniform(0, 1) < 0.3:
                ImageDraw.Draw(bcg_img).rectangle(
                    [0, 0, im_width, im_height*random.uniform(0.01, 0.2)], fill=0)
                ImageDraw.Draw(bcg_img).rectangle(
                    [0, im_height * random.uniform(0.8, 0.99), im_width, im_height], fill=0)

            # time code text
            ImageDraw.Draw(bcg_img).text((x, y), tc, font=font, fill=(color,color,color))

            # bbox = [float(x + t_width/2)/im_width,
            #         float(y + t_height*1.15/2)/im_height,
            #         float(t_width)/im_width,
            #         float(t_height*1.15)/im_height]

            bbox = [float(x * random.uniform(0.996,1.01) + t_width*1.1/ 2) / im_width,
                    float(y * random.uniform(0.998,1.02) + t_height * 1.15 / 2) / im_height,
                    float(t_width * 1.20) / im_width,
                    float(t_height * 1.20 ) / im_height]

            false_bbox = [min(abs(float(1 - x * random.uniform(0.7,1.5) + t_width*1.1/ 2) / im_width), 0.95),
                    min(abs(float(1 - y * random.uniform(0.7,1.5) + t_height * 1.15 / 2) / im_height), 0.95),
                    float(t_width * 1.20) / im_width,
                    float(t_height * 1.20 ) / im_height]


            sigma = random.uniform(0, 0.9)
            bcg_img = gaussian_filter(bcg_img, sigma)


            bcg_img = np.array(bcg_img, dtype='uint8')



            # imsave('tc_tmp/%s_%s_%s.jpg' % (tc, x, y), bcg_img)
            # cv2.imwrite('res/%s_%s_%s.jpg' % (tc, x, y), bcg_img)

            true_crop = cropBBox(bcg_img, bbox)
            false_crop = cropBBox(bcg_img, false_bbox)

            # cv2.imwrite('tmp/%s_%s_%s_true_crop.jpg' % (tc, x, y), true_crop)
            # cv2.imwrite('tmp/%s_%s_%s_false_crop.jpg' % (tc, x, y), false_crop)
            #cv2.imwrite('res/%s_%s_%s_bbox.jpg' % (tc, x, y), drawBBox(bbox, bcg_img))
            # mask = np.packbits(np.asarray(mask, dtype='bool'), axis=-1)
            return [true_crop,false_crop, tc]


pool = ThreadPool(max(cpu_count() // 2, 4))


def getLabel(s):
    label = np.zeros((tc_len, len(vocabulary)))
    for i, char in enumerate(s):
        label[i][vocabulary.index(char)] = 1
    return label


def gen(batch_size=32):
    x = np.zeros((batch_size, 27, 27*7, 3), dtype=np.float)
    y = np.zeros((batch_size, 11, len(vocabulary)))
    y_conf = np.zeros((batch_size, 1))
    while True:
        result = pool.map(process, range(batch_size/2))
        k = 0
        for i, v in enumerate(result):
            x[k] = v[0].astype(np.float) / 127.5 - 1
            y[k] = getLabel(v[2])
            y_conf[k] = 1
            k+=1

            x[k] = v[1].astype(np.float) / 127.5 - 1
            y[k] = 1./12.
            y_conf[k] = 0
            k+=1

        yield (x, [y,y_conf])

model = timecode_ocr_model_RGB_full()
model.load_weights('checkpoints/timecode_ocr_model_RGB_full_vl0.0649.hdf5')
model.summary()

print(len(fonts), len(bcgs))

model.fit_generator(generator=gen(batch_size=BATCH_SIZE),
                    validation_data=gen(batch_size=BATCH_SIZE),
                    steps_per_epoch=1000,
                    validation_steps=100,
                    epochs=nb_epoch,
                    max_q_size=100,
                    callbacks=[
                        callbacks.ModelCheckpoint(
                            'checkpoints/timecode_ocr_model_RGB_full_vl{val_loss:.4f}.hdf5',
                            monitor='val_loss')
                    ])
