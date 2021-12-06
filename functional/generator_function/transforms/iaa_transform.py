from abc import abstractmethod

import numpy as np
from imgaug import augmenters as iaa

from functional.generator_function.global_definition import ImageTransforms


class IAAImageTransform(ImageTransforms):
    """
    The implementation for iaa(imgaug) in transform
    """

    def __init__(self, threshold: float) -> None:
        """
        Init, threshold for the possible to perform this transform
        :param threshold: float, the possible to perform this transform
        """
        super().__init__(threshold)
        self.transform = None

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __call__(self, image: np.ndarray, label: np.ndarray) -> (np.ndarray, np.ndarray):
        pass

    def basic_transform(self, image: np.ndarray, label: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Basic transform, which means the label not changed during the transform
        :param image: np.ndarray, the image
        :param label: np.ndarray, the label in one-hot mode
        :return: the transformed image and label
        """
        # Check possibility
        if self.check_threshold():
            image = self.transform.augment_image(image)
            return image, label
        else:
            return image, label


class IAAGaussianBlur(IAAImageTransform):
    """
    Gaussian Blur implementation in iaa
    """

    def __init__(self, threshold: float, **kwargs):
        super().__init__(threshold)
        self.kwargs = kwargs
        self.transform = iaa.GaussianBlur(**self.kwargs)

    def __call__(self, image: np.ndarray, label: np.ndarray) -> (np.ndarray, np.ndarray):
        return self.basic_transform(image, label)

    def __str__(self):
        return "IAA.GaussianBlur: {}, {}".format(self.threshold, self.kwargs)
