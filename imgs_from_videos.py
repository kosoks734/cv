import os
import argparse
import cv2 as cv

def imgs_from_videos(input, output):
    dir = os.listdir(input)
    writer_count = 0
    for file in dir:
        vs = cv.VideoCapture(os.path.join(input, file))
        count = 0

        while True:
            ret, img = vs.read()

            if img is None:
                break

            if count % 15 == 0:
                cv.imwrite(os.path.join(output, "img_%i.jpeg" % writer_count), img)
                writer_count += 1

            count += 1
        vs.release()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, required=True,  help="path to input directory")
    ap.add_argument("-o", "--output", type=str, required=True,  help="path to output directory")
    args = vars(ap.parse_args())
    imgs_from_videos(args['input'], args['output'])
