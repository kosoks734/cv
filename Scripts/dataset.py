import os
import cv2 as cv
import numpy as np

class Dataset:
    """
    This class manages the information for the dataset.
    """
    def __init__(self):
        """
        Initialize the Dataset object.

        Args:
            void

        Returns:
            void
        """
        self.paths = []
        self.class_num = 0

    def add(self, path):
        """
        Gets path to directory with images and add it to list of paths.

        Args:
            path (string): Path to directory with images of class.

        Returns:
            void
        """
        self.paths.append(path)
        self.class_num += 1

    def remove(self, path):
        """
        Gets path to directory with images and remove it to list of paths.

        Args:
            path (string): Path to directory with images of class.

        Returns:
            void
        """
        self.paths.remove(path)
        self.class_num -= 1

    def load(self):
        """
        Loads images from all written paths.

        Args:
            void

        Returns:
            list of numpy float matrices: list of images from all classes.
            list of integers: list of labels of images.
        """
        data, labels = [], []

        for idx, path in enumerate(self.paths):
            for filename in os.listdir(path):
                img = cv.imread(os.path.join(path, filename), cv.IMREAD_GRAYSCALE)

                if img is not None:
                    data.append(img)
                    labels.append(self.class_num - idx - 1)

        return data, np.array(labels)
