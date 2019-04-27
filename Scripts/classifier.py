import os
import cv2 as cv
import numpy as np

from Scripts.dataset import Dataset
from Scripts.bow import BoW

class Classifier:
    """
    Class for making training and testing in image classification.
    """
    def __init__(self, bow):
        """
        Initialize the Classifier object.

        Args:
            bow (BoW): The object that stores BoW descriptor.

        Returns:
            void
        """
        self.bow = bow
        self.svm = cv.ml.SVM_create()

    def save(self, path):
        """
        Gets path and saves the SVM there.

        Args:
            path (string): Path where needs to save.

        Returns:
            void
        """
        self.svm.save(path)

    def load(self, path):
        """
        Gets path and loads the SVM from there.

        Args:
            path (string): Path where SVM located.

        Returns:
            void
        """
        self.svm.load(path)

    def train(self, dataset):
        """
        Gets dataset and train SVM on this

        Args:
            dataset (Dataset): The object that stores the information about the dataset.

        Returns:
            void
        """
        traindata, trainlabels = dataset.load()

        if self.bow is not None:
            traindata = self.bow.extract(traindata)

        self.svm.trainAuto(np.array(traindata), cv.ml.ROW_SAMPLE, np.array(trainlabels))

    def predict(self, dataset):
        """
        Gets dataset and predict it with trained SVM

        Args:
            dataset (Dataset): The object that stores the information about the dataset.

        Returns:
            void
        """
        testdata = dataset.load()[0]

        if self.bow is not None:
            testdata = self.bow.extract(testdata)

        return self.svm.predict(np.array(testdata))
