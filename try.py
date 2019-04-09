import os
import cv2 as cv
import numpy as np

def get_data(path):
    images=[]
    dirs=os.listdir(path)

    for filename in dirs:
        img=cv.imread(os.path.join(path, filename), cv.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

    return images

#Step 1
#Compute decriptors of part of training set by SIFT and train BoW by KMeans

detector = cv.xfeatures2d.SIFT_create()
extractor = cv.xfeatures2d.SIFT_create()

train_bow = get_data("/home/andreeff/Pictures/BOW")

bow_trainer = cv.BOWKMeansTrainer(17 ** 3)

for img in train_bow:
    bow_trainer.add(extractor.compute(img, detector.detect(img))[1])

bow = bow_trainer.cluster()

#Step 2
#Compute descriptors of training set by SIFT+BoW and training SVM model

train_pos = get_data("/home/andreeff/Pictures/Train/Pos")
train_neg = get_data("/home/andreeff/Pictures/Train/Neg")

flann_params = dict(algorithm = 1, trees = 5)
matcher = cv.FlannBasedMatcher(flann_params, {})

bow_extractor = cv.BOWImgDescriptorExtractor(extractor, matcher)
bow_extractor.setVocabulary(bow)

traindata, trainlabels = [], []

for img in train_pos:
    traindata.extend(bow_extractor.compute(img, detector.detect(img))); trainlabels.append(1)

for img in train_neg:
    traindata.extend(bow_extractor.compute(img, detector.detect(img))); trainlabels.append(-1)

svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_NU_SVC)
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setGamma(2)
svm.setNu(0.05)
svm.setDegree(3)
svm.train(np.array(traindata), cv.ml.ROW_SAMPLE, np.array(trainlabels))

test_pos_old = get_data("/home/andreeff/Pictures/Test/Pos/Old")
test_neg_old = get_data("/home/andreeff/Pictures/Test/Neg/Old")
test_pos_new = get_data("/home/andreeff/Pictures/Test/Pos/New")
test_neg_new = get_data("/home/andreeff/Pictures/Test/Neg/New")

for img in test_pos_old:
    print(svm.predict(bow_extractor.compute(img, detector.detect(img)))[1][0][0])

print("--------")

for img in test_neg_old:
    print(svm.predict(bow_extractor.compute(img, detector.detect(img)))[1][0][0])

print(" ")
print("--------")

for img in test_pos_new:
    print(svm.predict(bow_extractor.compute(img, detector.detect(img)))[1][0][0])

print("--------")

for img in test_neg_new:
    print(svm.predict(bow_extractor.compute(img, detector.detect(img)))[1][0][0])