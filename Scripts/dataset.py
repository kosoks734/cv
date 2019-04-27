import os
import cv2 as cv

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
        self.classes = {}

    def add(self, path, class_num):
        """
        Gets path to directory with images and add it to list of paths.

        Args:
            path (string): Path to directory with images of class.

        Returns:
            void
        """
        if self.classes.get(class_num) is None:
            self.classes[class_num] = []

        self.classes[class_num].append(path)

    def remove(self, path, class_num):
        """
        Gets path to directory with images and remove it to list of paths.

        Args:
            path (string): Path to directory with images of class.

        Returns:
            void
        """
        if self.classes.get(class_num) is None:
            return

        self.classes[class_num].remove(path)

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

        for key in self.classes:
            for path in self.classes[key]:
                for filename in os.listdir(path):
                    img = cv.imread(os.path.join(path, filename), cv.IMREAD_GRAYSCALE)

                    if img is not None:
                        data.append(img)
                        labels.append(key)

        return data, labels
