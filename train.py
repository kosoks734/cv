import argparse
import numpy as np
import cv2
import data

def train(pos, neg):
    bow, det=data.get_bow_extractor()
    traindata, trainlabels=[], []
    for img in pos:
        traindata.extend(bow.compute(img, det.detect(img)))
        trainlabels.append(1.)
    print("Training SVM...")
    svm=cv2.ml.SVM_create()
    svm.setNu(0.0003)
    svm.setType(cv2.ml.SVM_ONE_CLASS)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.train(np.float32(traindata), cv2.ml.ROW_SAMPLE, np.int32(trainlabels))
    print("Done")
    print("Saving SVM...")
    svm.save("svm.xml")
    print("Done")
    
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-p", "--pos", type=str, required=True,  help="path to img directory")
    ap.add_argument("-n", "--neg", type=str, required=True,  help="path to img directory")
    args=vars(ap.parse_args())
    pos=data.get_data(args["pos"])
    neg=data.get_data(args["neg"])
    train(pos, neg)
