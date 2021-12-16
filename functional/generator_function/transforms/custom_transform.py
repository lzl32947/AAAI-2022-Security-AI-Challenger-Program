from abc import abstractmethod

import numpy as np

from functional.generator_function.global_definition import ImageTransforms


class CustomImageTransform(ImageTransforms):
    """
    The implementation for iaa(imgaug) in transform
    """

    def __init__(self, threshold: float) -> None:
        """
        Init, threshold for the possible to perform this transform
        :param threshold: float, the possible to perform this transform
        """
        super().__init__(threshold)

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __call__(self, image: np.ndarray, label: np.ndarray) -> (np.ndarray, np.ndarray):
        pass



class Cutup(CustomImageTransform):
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
