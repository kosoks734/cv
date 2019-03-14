import cv2
import numpy as np
import argparse
import time

ap=argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,  help="path to input video (if empty then used webcam)")
ap.add_argument("-o", "--output", type=str, help="path to output video (optional)")
ap.add_argument("-f", "--skip-frames", type=int, default=1, help="number of skip frames between detections")
ap.add_argument("-w", "--win-stride", type=str, default="(4, 4)", help="window stride")
ap.add_argument("-s", "--scale", type=float, default=1.05, help="image pyramid scale")
ap.add_argument("-r", "--resize", type=float, default=1, help="resize param")
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
prev=False
count=0

while True:
    ret, img=vs.read()

    if args["input"] is not None and img is None:
        break

    frame_width=int(vs.get(3))
    frame_height=int(vs.get(4))

    if args["output"] is not None and writer_pos is None:
        fourcc=cv2.VideoWriter_fourcc(*"XVID")
        writer_pos=cv2.VideoWriter(args["output"] + "det.avi", fourcc, vs.get(cv2.CAP_PROP_FPS), (frame_width, frame_height), True)

    if args["output"] is not None and writer_neg is None:
        fourcc=cv2.VideoWriter_fourcc(*"XVID")
        writer_neg=cv2.VideoWriter(args["output"] + "ndet.avi", fourcc, vs.get(cv2.CAP_PROP_FPS), (frame_width, frame_height), True)

    if count % args["skip_frames"] == 0:
        frame=cv2.resize(img, (0,0), fx=args["resize"], fy=args["resize"])
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        (rects, weights)=hog.detectMultiScale(frame, winStride=win_stride, padding=(8,8), scale=args["scale"])
   
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not len(rects) == 0 and writer_pos is not None:
            writer_pos.write(frame)
            prev=True
        elif len(rects) == 0 and writer_neg is not None:
            writer_neg.write(frame)
            prev=False
    else:
        frame=cv2.resize(img, (0,0), fx=args["resize"], fy=args["resize"])

        if prev:
            writer_pos.write(frame)
        else:
            writer_neg.write(frame)

    count+=1
    cv2.imshow("frame", frame)
    cv2.waitKey(10)

if writer_pos is not None:
    writer_pos.release()
if writer_neg is not None:
    writer_neg.release()
if not args.get("input", False):
    vs.stop()
else:
    vs.release()

cv2.destroyAllWindows()