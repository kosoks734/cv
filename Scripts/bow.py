import os
import cv2 as cv
import numpy as np

class BoW:
    """
    Class for training Bag of Features and computing descriptors.
    """
    def __init__(self, des_type):
        """
        Initialize the Bag of Features object.

        Args:
            des_type (string): Type of image descriptor SIFT or ORB.

        Returns:
            void
        """
        self.des_type = des_type
        self.extractor = cv.xfeatures2d.SIFT_create() if des_type == "SIFT" else cv.ORB_create()
        self.bow = None

    def train(self, imgs):
        """
        Gets image list and cluster Bag of Words.

        Args:
            images (list of numpy float matrices): List of images for train BoF.

        Returns:
            void
        """
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        bow_trainer = cv.BOWKMeansTrainer(1000, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

        for img in imgs:
            des = self.extractor.detectAndCompute(img, None)[1]

            if self.des_type != "SIFT":
                des = np.float32(des)

            bow_trainer.add(des)

        self.bow = bow_trainer.cluster()

    def extract(self, imgs):
        """
        Gets list of images and compute descriptors for them.

        Args:
            images (list of numpy float matrices): Set of images for calculate the descriptor.

        Returns:
            Numpy array: Descriptors of given set of images.
        """
        if self.des_type == "SIFT":
            flann_params = dict(algorithm=1, trees=5)
            matcher = cv.FlannBasedMatcher(flann_params, {})
        else:
            matcher = cv.DescriptorMatcher.create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)

        bow_extractor = cv.BOWImgDescriptorExtractor(self.extractor, matcher)

        if self.des_type != "SIFT":
            bow_extractor.setVocabulary(np.uint8(self.bow))
        else:
            bow_extractor.setVocabulary(self.bow)

        des = []

        for img in imgs:
            des.extend(bow_extractor.compute(img, self.extractor.detect(img)))

        return np.array(des)

    def save(self, path):
        """
        Gets path and saves the BoF there.

        Args:
            bow (numpy float matrix): The Bag of Features.
            path (string): Path to directory where needs to save.

        Returns:
            void
        """
        file_storage = cv.FileStorage(os.path.join(path, "BoF.xml"), cv.FILE_STORAGE_WRITE)
        file_storage.write("bof", self.bow)
        file_storage.release()

    def load(self, path):
        """
        Gets path and loads the BoW from there.

        Args:
            path (string): Path to directory with BoF.

        Returns:
            void
        """
        file_storage = cv.FileStorage(path, cv.FILE_STORAGE_READ)
        self.bow = file_storage.getNode("bof").mat()
        file_storage.release()
