import cv2
import numpy as np
import argparse
import time

ap=argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,  help="path to input video (if empty then used webcam)")
ap.add_argument("-o", "--output", type=str, help="path to output video (optional)")
#ap.add_argument("-s", "--skip-frames", type=int, default=30, help="number of skip frames between detections") can be used for faster detect if use tacking
ap.add_argument("-w", "--win-stride", type=str, default="(8, 8)", help="window stride")
ap.add_argument("-p", "--padding", type=str, default="(16, 16)", help="object padding")
ap.add_argument("-s", "--scale", type=float, default=1.05, help="image pyramid scale")
ap.add_argument("-t", "--hit-threshold", type=float, default=2.0, help="threshold for the distance between features and svm classifying plane")
ap.add_argument("-m", "--mean-shift", type=int, default=-1, help="whether or not mean shift grouping should be used")
args=vars(ap.parse_args())
if not args.get("input", False):
    vs=cv2.VideoCapture(0)
    time.sleep(2.0)
else:
    vs=cv2.VideoCapture(args["input"])
hog=cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
writer_pos=None
writer_neg=None
win_stride=eval(args["win_stride"])
padding=eval(args["padding"])
mean_shift=True if args["mean_shift"] > 0 else False
while True:
    ret, img=vs.read()
    if args["input"] is not None and img is None:
        break
    frame=cv2.resize(img, (200,200))
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_width=int(vs.get(3))
    frame_height=int(vs.get(4))
    if args["output"] is not None and writer_pos is None:
        fourcc=cv2.VideoWriter_fourcc(*"XVID")
        writer_pos=cv2.VideoWriter(args["output"] + "det.avi", fourcc, vs.get(cv2.CAP_PROP_FPS), (200, 200), True)
    if args["output"] is not None and writer_neg is None:
        fourcc=cv2.VideoWriter_fourcc(*"XVID")
        writer_neg=cv2.VideoWriter(args["output"] + "ndet.avi", fourcc, vs.get(cv2.CAP_PROP_FPS), (200, 200), True)
    (rects, weights)=hog.detectMultiScale(frame, winStride=win_stride, padding=padding, scale=args["scale"], useMeanshiftGrouping=mean_shift, hitThreshold=args["hit_threshold"])
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)ак
    if not len(rects) == 0 and writer_pos is not None:
        writer_pos.write(frame)
    elif len(rects) == 0 and writer_neg is not None:
        writer_neg.write(frame)
    cv2.imshow("frame", frame)
if writer_pos is not None:
    writer_pos.release()
if writer_neg is not None:
    writer_neg.release()
if not args.get("input", False):
    vs.stop()
else:
    vs.release()




