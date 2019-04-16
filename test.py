import argparse
import numpy as np
import data

def test(images):
    print("Start test...")
    svm=data.get_svm_classifier()
    bow, det=data.get_bow_extractor()
    traindata=[]
    for img in images:
        traindata.extend(bow.compute(img, det.detect(img)))
    res=svm.predict(np.float32(traindata))
    print(res)

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, required=True,  help="path to img directory")
    args=vars(ap.parse_args())
    images=data.get_data(args["input"])
    test(images)
