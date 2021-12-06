import random
from abc import abstractmethod, ABC
from typing import Union, List, Tuple, Optional

import numpy as np


class ImageTransforms:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def check_threshold(self):
        if random.random() < self.threshold:
            return True
        else:
            return False

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __call__(self, image, label):
        pass


class ImageCompose:
    def __init__(self, transforms: Optional[Union[List, Tuple, ImageTransforms]]):
        if transforms is None:
            self.transforms = []
        if isinstance(transforms, (list, tuple)):
            self.transforms = transforms
        elif isinstance(transforms, ImageTransforms):
            self.transforms = [transforms]
        else:
            self.transforms = []

    def __call__(self, image, label):
        if len(self.transforms) != 0:
            for t in self.transforms:
                image, label = t(image, label)
        return image, label

    def __str__(self):
        if self.transforms is None or len(self.transforms) == 0:
            return "Composed: None"
        else:
            s = "Composed: [\n"
            for item in self.transforms:
                s += str(item) + "\n"
            s += "]"
            return s


class ArgumentRunnable:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __call__(self, *args, **kwargs):
        data_list = []
        label_list = []
        for image, label in self.dataset:
            image = np.array(image, dtype=np.uint8)
            if isinstance(label, int):
                label = self.to_one_hot(label)
            label = np.array(label)
            image, label = self.transform(image, label)
            data_list.append(image)
            label_list.append(label)
        data_list = np.array(data_list)
        label_list = np.array(label_list)
        return data_list, label_list

    @staticmethod
    def to_one_hot(label):
        t = np.zeros(shape=(10,), dtype=np.float)
        t[label] = 1.0
        return t
