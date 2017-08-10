import numpy as np
import sys
import cv2 as cv2
import math
from copy import deepcopy
from scipy import stats


try:
    import Image
except ImportError:
    from PIL import Image

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def crop_minAreaRect(img, rect):

    # rotate img
    # if (rect[2]) < -60:
    #     angle = 115 + rect[2]
    # elif (rect[2]) > 60:
    #     angle = 115 + rect[2]
    # else:
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    cy = (rect[1][1] + rect[0][1]) / 2
    cx = (rect[1][0] + rect[0][0]) / 2
    M = cv2.getRotationMatrix2D((cx,cy),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]

    if img_crop.shape[0] > img_crop.shape[1] :
        img_crop = np.rot90(img_crop)

    return img_crop


def adjust(img, mask):
    # thresholding and inversion of image for further processing.
    # handy for controur extraction
    def threshinv(img):
        ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
        return thresh

    # dilation over a thresholded image would cause any word to pixelate into one contour
    def dilat(img):
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return result

    mask = dilat(threshinv(mask))
    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    xmax = 0
    xmin = 100000
    ymin = 100000
    ymax = 0
    havg = 0
    thetaavg = 0
    total_a = 0
    maxAreaBox = None
    maxArea = 0
    allCnt = None
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        a = img.size / max(0.001, cnt_area)

        if a > 2 and a < 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            if x < xmin:
                xmin = x
            if y < ymin:
                ymin = y
            if x + w > xmax:
                xmax = x + w
            if y + h > ymax:
                ymax = y + h
            havg = havg + h*a
            rect = cv2.minAreaRect(cnt)

            if cnt_area > maxArea:
                maxAreaBox = deepcopy(rect)
                maxArea = cnt_area

            if type(allCnt) is np.ndarray:
                allCnt = np.concatenate((allCnt, cnt))
            else:
                allCnt = cnt.copy()

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            mask = cv2.drawContours(mask, [box], 0, (180, 180, 180), 2)

            # assuming that user does not rotate more than 45 degrees
            total_a += a

    # havg is the average height of text
    # similar method can be used for getting average theta(using minimum bounding box along with boundingbox) for rotation and then rotating accordingly
    if maxAreaBox != None:
        box = cv2.boxPoints(maxAreaBox)
        box = np.int0(box)

        mask = cv2.drawContours(mask, [box], 0, (127, 127, 127), 3)
        height = np.linalg.norm(box[1]-box[0])

        rows, cols = img.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(allCnt, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)

        print(cols-1, righty, lefty)

        mask = cv2.line(mask, (cols - 1, righty), (0, lefty), (127, 127, 127), 2)

        a = np.array([cols - 1, righty])
        b = np.array([0, lefty])
        nearPoints = []
        coords = np.array(np.where(mask < 127)).transpose()[:, ::-1]
        for p in coords:
            d = np.linalg.norm(np.cross(b - a, a - p)) / np.linalg.norm(b - a)
            if d <= height/2.:
                nearPoints.append(p)
        nearPoints = np.array(nearPoints)
        if len(nearPoints) > 2:
            [vx, vy, x, y] = cv2.fitLine(nearPoints, cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
        else:
            nearPoints = coords
        mask = cv2.line(mask, (cols - 1, righty), (0, lefty), (0, 0, 0), 2)

        minrect = cv2.minAreaRect(nearPoints)
        box = cv2.boxPoints(minrect)
        box = np.int0(box)
        mask = cv2.drawContours(mask, [box], 0, (0, 0, 0), 2)
        output = four_point_transform(img, box)
        # cv2.imwrite('tempmask.png', mask)
        # cv2.imwrite('tempcut.png', output)

        return output, mask
    else:
        return img, mask


def adjust_horizontal(img, mask):
    # thresholding and inversion of image for further processing.
    # handy for controur extraction
    def threshinv(img):
        ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
        return thresh

    # dilation over a thresholded image would cause any word to pixelate into one contour
    def dilat(img):
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return result

    # cv2.imwrite('tmp_ormask.jpg', mask)

    mask = dilat(threshinv(mask))
    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    xmax = 0
    xmin = 100000
    ymin = 100000
    ymax = 0
    havg = 0
    maxAreaBox = None
    maxArea = 0
    allCnt = None
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        a = img.size / max(0.001, cnt_area)

        if (a > 2 and a < 3000):
            x, y, w, h = cv2.boundingRect(cnt)
            if x < xmin:
                xmin = x
            if y < ymin:
                ymin = y
            if x + w > xmax:
                xmax = x + w
            if y + h > ymax:
                ymax = y + h
            havg = havg + h * a
            rect = cv2.minAreaRect(cnt)

            if cnt_area > maxArea:
                maxAreaBox = deepcopy(rect)
                maxArea = cnt_area

            if type(allCnt) is np.ndarray:
                allCnt = np.concatenate((allCnt, cnt))
            else:
                allCnt = cnt.copy()

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            mask = cv2.drawContours(mask, [box], 0, (180, 180, 180), 2)

    # havg is the average height of text
    # similar method can be used for getting average theta(using minimum bounding box along with boundingbox) for rotation and then rotating accordingly
    if maxAreaBox != None:
        box = cv2.boxPoints(maxAreaBox)
        box = np.int0(box)

        # mask = cv2.drawContours(mask, [box], 0, (127, 127, 127), 3)
        height = np.linalg.norm(box[1] - box[0])

        y = np.mean(allCnt[:, :, 1])

        nearPoints = []
        coords = np.array(np.where(mask < 127)).transpose()[:, ::-1]


        for p in coords:
            d = abs(p[1] - y)
            if d <= height / 2.:
                nearPoints.append(p)
        nearPoints = np.array(nearPoints, dtype='int32')

        try:
            min_x = max(0, np.min(nearPoints[:, 0]) - 5)
            max_x = np.max(nearPoints[:, 0]) + 5
            min_y = max(0, np.min(nearPoints[:, 1]) - 5)
            max_y = np.max(nearPoints[:, 1]) + 5
            output = img[min_y: max_y, min_x: max_x]
        except:
            print('shinema huinya', nearPoints.shape)
            return img, mask

        cv2.imwrite('tmp_img.jpg', img[:, :])
        cv2.imwrite('tmp.jpg', output[:, :])
        cv2.imwrite('tmp_mask.jpg', mask)

        return output, mask
    else:
        return img, mask

# img = cv2.imread('tmp_img.jpg')[:, :, 0:1]
# mask = cv2.imread('tmp_ormask.jpg')[:, :, 0:1]
#
# adjust_horizontal(img, mask)
