import argparse
import numpy as np
import cv2
import data

def train(images):
    bow, det=data.get_bow_extractor()
    traindata, trainlabels=[], []
    for img in images:
        traindata.extend(bow.compute(img, det.detect(img)))
        trainlabels.append(1.)
    print("Training SVM...")
    svm=cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_ONE_CLASS)
    svm.setNu(0.05)
    svm.setDegree(3)
    svm.train(np.float32(traindata), cv2.ml.ROW_SAMPLE, np.int32(trainlabels))
    print("Done")
    print("Saving SVM...")
    svm.save("svm.xml")
    print("Done")
    
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, required=True,  help="path to img directory")
    args=vars(ap.parse_args())
    images=data.get_data(args["input"])
    train(images)
