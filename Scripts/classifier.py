import os
import cv2 as cv
import numpy as np

from dataset import Dataset
from bow import BoW

class Classifier:
    """
    Class for making training and testing in image classification.
    """
    def __init__(self, bow: BoW):
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
            path (string): Path to directory where needs to save.

        Returns:
            void
        """
        self.svm.save(os.path.join(path, "SVM.xml"))

    def load(self, path):
        """
        Gets path and loads the SVM from there.

        Args:
            path (string): Path to directory with SVM.

        Returns:
            void
        """
        self.svm.load(path)

    def train(self, dataset: Dataset):
        """
        Gets dataset and train SVM on this

        Args:
            dataset (Dataset): The object that stores the information about the dataset.
        
        Returns:
            void
        """
        traindata, trainlabels = dataset.load()
        traindata = self.bow.extract(traindata)
        self.svm.train(traindata, cv.ml.ROW_SAMPLE, trainlabels)

    def predict(self, dataset: Dataset):
        """
        Gets dataset and predict it with trained SVM

        Args:
            dataset (Dataset): The object that stores the information about the dataset.

        Returns:
            void
        """
        testdata = dataset.load()[0]
        testdata = self.bow.extract(testdata)
        return self.svm.predict(testdata)
