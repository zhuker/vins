import numpy as np
import cv2
import os
timecode_cascade = cv2.CascadeClassifier('models/cascade.xml')

rgbs = [os.path.join(root, f) for root, _, files in os.walk('input') for f in files if f.endswith('.jpg')]
for imgpath in rgbs:
    img = cv2.resize(cv2.imread(imgpath), (512,302))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    timecodes = np.ones((1,4), dtype=np.int)
    t = timecode_cascade.detectMultiScale(gray, 3.5, 1)
    if len(t) > 0:
        timecodes = np.concatenate((t, timecodes))
    t = timecode_cascade.detectMultiScale(gray, 3., 9)
    if len(t) > 0:
        timecodes = np.concatenate((t, timecodes))
    t = timecode_cascade.detectMultiScale(gray, 5., 10)
    if len(t) > 0:
        timecodes = np.concatenate((t, timecodes))
    t = timecode_cascade.detectMultiScale(gray, 2., 10)
    if len(t) > 0:
        timecodes = np.concatenate((t, timecodes))
    t = timecode_cascade.detectMultiScale(gray, 1.01, 10)
    if len(t) > 0:
        timecodes = np.concatenate((t, timecodes))

    for (x, y, w, h) in timecodes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

    cv2.imshow('img', img)
    k = cv2.waitKey(500) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()