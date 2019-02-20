import numpy as np
import cv2
import argparse

ag=argparse.ArgumentParser()
ag.add_argument("-i", "--input", type=str)
ag.add_argument("-w", "--win-stride", type=str, default="(8, 8)")
ag.add_argument("-p", "--padding", type=str, default="(16, 16)")
ag.add_argument("-s", "--scale", type=float, default=1.03)
args=vars(ag.parse_args())
hog=cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
win_stride=eval(args["win_stride"])
padding=eval(args["padding"])
img=cv2.imread(args["input"], 0)
img=cv2.resize(img, (295,295))
(rects, weights)=hog.detectMultiScale(img, winStride=win_stride, padding=padding, scale=args["scale"])
for (x, y, w, h) in rects:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("sobaka", img)
cv2.waitKey(2000)
cv2.destroyAllWindows()

