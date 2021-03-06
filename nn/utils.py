import cv2
import math
import random

vocabulary = ' 1234567890ABCDEFGHJKLMNPRSTUVWXYZ'


def generateVin(vin_len, max_spaces):
    vin = ''
    spaces = 0
    while len(vin) < vin_len:
        c = random.choice(vocabulary)
        if c == ' ' and spaces < max_spaces:
            spaces += 1
        elif c == ' ' and spaces >= max_spaces:
            continue
        vin += c
    return vin.strip()


def generate_timecode():
    s = ('%02d:%02d:%02d' + random.choice(':;') + '%02d') % \
        (random.randint(0, 99), random.randint(0, 59), random.randint(0, 59), random.randint(0, 99))

    return s


def getLabel(vin):
    Y = [vocabulary.index(char) for char in vin]
    return Y, len(vin)


def mirror(a,b,angle):
    c = ((a[0] + b[0])/2, (a[1] + b[1])/2)
    a_r = rotate_coord(a[0], a[1], c[0], c[1], -angle)
    b_r = rotate_coord(b[0], b[1], c[0], c[1], -angle)

    a_rr = rotate_coord(a_r[0], b_r[1], c[0], c[1], angle)
    b_rr = rotate_coord(b_r[0], a_r[1], c[0], c[1], angle)

    return a_rr, b_rr

# rotate point (x, y) around (x0, y0) on angle
# taking into account that Y axis looks down
def rotate_coord(x, y, x0, y0, angle):
    rad = math.radians(angle)
    x -= x0
    y -= y0
    xr = x * math.cos(rad) + y * math.sin(rad) + x0
    yr = y * math.cos(rad) - x * math.sin(rad) + y0
    return [xr, yr]


def crop_rotate(image, angle, a, b):
    (h, w) = image.shape[:2]

    center = ((a[0]+b[0])/2., (a[1]+b[1])/2.)

    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))

    a_rotated = rotate_coord(a[0], a[1], center[0], center[1], angle)
    b_rotated = rotate_coord(b[0], b[1], center[0], center[1], angle)

    minx = int(min(a_rotated[0], b_rotated[0], w))
    miny = int(min(a_rotated[1], b_rotated[1], h))
    maxx = int(max(a_rotated[0], b_rotated[0], 0))
    maxy = int(max(a_rotated[1], b_rotated[1], 0))

    crop = rotated[miny:maxy, minx:maxx]
    return crop

def drawBBox(bbox, img):
    tl_x = int((bbox[0]-(bbox[2]/2))*img.shape[1])
    tl_y = int((bbox[1]-(bbox[3]/2))*img.shape[0])
    br_x = int((bbox[0]+(bbox[2]/2))*img.shape[1])
    br_y = int((bbox[1]+(bbox[3]/2))*img.shape[0])

    cv2.rectangle(img, (tl_x, tl_y), (br_x, br_y), (255,0,0), thickness=1, lineType=8, shift=0)
    return img