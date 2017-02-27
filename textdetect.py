
import sys
import os

import cv2
import numpy as np

pathname = '.'


img      = cv2.imread('./IMG_0233.JPG')
#img = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
# for visualization
vis      = img.copy()


# Extract channels to be processed individually
channels = cv2.text.computeNMChannels(img)
# Append negative channels to detect ER- (bright regions over dark background)
cn = len(channels)-1
for c in range(0,cn):
  channels.append((255-channels[c]))

# Apply the default cascade classifier to each independent channel (could be done in parallel)
print("Extracting Class Specific Extremal Regions from "+str(len(channels))+" channels ...")
print("    (...) this may take a while (...)")
for channel in channels:

  erc1 = cv2.text.loadClassifierNM1(pathname+'/trained_classifierNM1.xml')
  er1 = cv2.text.createERFilterNM1(erc1,16,0.00015,0.13,0.2,True,0.1)

  erc2 = cv2.text.loadClassifierNM2(pathname+'/trained_classifierNM2.xml')
  er2 = cv2.text.createERFilterNM2(erc2,0.5)

  regions = cv2.text.detectRegions(channel,er1,er2)

  rects = cv2.text.erGrouping(img,channel,[r.tolist() for r in regions])
  #rects = cv2.text.erGrouping(img,channel,[x.tolist() for x in regions], cv2.text.ERGROUPING_ORIENTATION_ANY,'./trained_classifier_erGrouping.xml',0.5)

  #Visualization
  for r in range(0,np.shape(rects)[0]):
    rect = rects[r]
    cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
    cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)


#Visualization
cv2.imshow("Text detection result", vis)
cv2.waitKey(0)
