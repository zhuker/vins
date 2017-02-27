
import cv2
import numpy as np

pathname = '.'


img      = cv2.imread('./IMG_0233.JPG')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = min(img.shape[0], img.shape[1])
maxLineGap = 10
lines = cv2.HoughLines(edges,1,np.pi/180,132,minLineLength,maxLineGap)
for line in lines:
    print line
    rho = line[0][0]
    theta = line[0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow("asdf", img)
cv2.waitKey(0)
