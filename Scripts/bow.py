import os
import cv2 as cv
import numpy as np

class BoW:
    """
    Class for training Bag of Features and computing descriptors.
    """
    def __init__(self, des_name):
        """
        Initialize the Bag of Features object.

        Args:
            des_type (string): Type of image descriptor SIFT or ORB.

        Returns:
            void
        """
        des_names = {
            'BRISK': cv.BRISK_create,
            'ORB': cv.ORB_create,
            'AKAZE': cv.AKAZE_create,
            'KAZE': cv.KAZE_create,
            'SIFT': cv.xfeatures2d.SIFT_create,
            'SURF': cv.xfeatures2d.SURF_create
        }
        self.detector = des_names[des_name]()
        self.extractor = des_names[des_name]()
        self.bow = None

    def train(self, imgs, n_clusters):
        """
        Gets image list and cluster Bag of Words.

        Args:
            images (list of numpy float matrices): List of images for train BoF.

        Returns:
            void
        """
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        bow_trainer = cv.BOWKMeansTrainer(n_clusters, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

        for img in imgs:
            des = self.extractor.compute(img, self.detector.detect(img))[1]
            bow_trainer.add(np.float32(des) if self.detector.descriptorType() == 0 else des)

        self.bow = bow_trainer.cluster()

    def extract(self, imgs):
        """
        Gets list of images and compute descriptors for them.

        Args:
            images (list of numpy float matrices): Set of images for calculate the descriptor.

        Returns:
            Numpy array: Descriptors of given set of images.
        """
        matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING if self.detector.descriptorType() == 0 else cv.DESCRIPTOR_MATCHER_FLANNBASED)
        bow_extractor = cv.BOWImgDescriptorExtractor(self.extractor, matcher)
        bow_extractor.setVocabulary(np.uint8(self.bow) if self.detector.descriptorType() == 0 else self.bow)

        des = []

        for img in imgs:
            kp = self.detector.detect(img)
            hist = bow_extractor.compute(img, kp)
            des.extend(hist)

        return des

    def save(self, path):
        """
        Gets path and saves the BoF there.

        Args:
            bow (numpy float matrix): The Bag of Features.
            path (string): Path where needs to save.

        Returns:
            void
        """
        file_storage = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
        file_storage.write("bof", self.bow)
        file_storage.release()

    def load(self, path):
        """
        Gets path and loads the BoW from there.

        Args:
            path (string): Path where BoF located.

        Returns:
            void
        """
        file_storage = cv.FileStorage(path, cv.FILE_STORAGE_READ)
        self.bow = file_storage.getNode("bof").mat()
        file_storage.release()
