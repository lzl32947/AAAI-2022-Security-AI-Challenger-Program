from abc import abstractmethod
from typing import Tuple

import numpy as np

from functional.generator_function.global_definition import ImageTransforms


class VerticalCutup(ImageTransforms):
    """
    Gaussian Blur implementation in
    """

    def __init__(self, threshold: float, **kwargs):
        super().__init__(threshold)
        self.kwargs = kwargs

    def __call__(self, image: np.ndarray, label: np.ndarray, *args) -> Tuple[np.ndarray, np.ndarray]:
        extra_image, extra_label = args[0], args[1]
        width = extra_image.shape[1] // 2
        image[:, width:,:] = extra_image[:, width:,:]
        label = (label + extra_label) / np.sum(label + extra_label)
        return image, label

    def __str__(self):
        return "VerticalCutup: {}, {}".format(self.threshold, self.kwargs)
