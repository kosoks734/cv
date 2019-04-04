import os
import numpy as np
import cv2

def get_data(path):
    images=[]
    print("Loading images from %s..." % path)
    dirs=os.listdir(path)

    for filename in dirs:
        img=cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

    print("Done")

    return images

def get_bow_extractor():
    print("Loading BoF...")
    fs=cv2.FileStorage("Vocabulare.xml", cv2.FILE_STORAGE_READ)
    voc=fs.getNode("voc").mat()
    fs.release()   
    matcher=cv2.DescriptorMatcher.create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    detector=cv2.ORB_create()
    extractor=cv2.ORB_create()
    bow_extractor=cv2.BOWImgDescriptorExtractor(extractor, matcher)
    bow_extractor.setVocabulary(np.uint8(voc))

    print("Done")

    return bow_extractor, detector

def get_svm_classifier():
    print("Loading SVM...")
    svm=cv2.ml_SVM.load("svm.xml")
    print("Done")
    return svm