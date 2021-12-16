import random
from abc import abstractmethod, ABC
from typing import Union, List, Tuple, Optional, Iterable, Sized
from tqdm import tqdm
import numpy as np

from functional.generator_function.dataset_function import RuntimeDataset


class ImageTransforms:
    """
    Father of all Image Transforms
    """

    def __init__(self, threshold: float) -> None:
        """
        Init the transform
        :param threshold: float, the possible to perform this transform
        """
        self.threshold = threshold

    def check_threshold(self) -> bool:
        """
        Check if to perform transformation this time
        :return: bool
        """
        if random.random() < self.threshold:
            return True
        else:
            return False

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __call__(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass


class ImageCompose:
    """
    Compose class to compose different transform sequentially
    """

    def __init__(self, transforms: Optional[Union[List, Tuple, ImageTransforms]]) -> None:
        """
        Init to compose
        :param transforms: List or Tuple or ImageTransforms, which should all be ImageTransforms
        """
        # Change it to list
        if transforms is None:
            self.transforms = []
        if isinstance(transforms, (list, tuple)):
            self.transforms = [i for i in transforms]
        elif isinstance(transforms, ImageTransforms):
            self.transforms = [transforms]
        else:
            self.transforms = []

    def __call__(self, image: np.ndarray, label: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.transforms) != 0:
            # Sequentially perform the transform
            for t in self.transforms:
                image, label = t(image, label, *args, **kwargs)
        return image, label

    def __str__(self) -> str:
        """
        Generate the description of this compose
        :return: str
        """
        if self.transforms is None or len(self.transforms) == 0:
            return "Composed: None"
        else:
            s = "Composed: [\n"
            for item in self.transforms:
                s += str(item) + "\n"
            s += "]"
            return s


class ArgumentRunnable:
    """
    The implementation of runnable of image argumentation
    """

    def __init__(self, dataset: RuntimeDataset, transform: ImageCompose) -> None:
        """
        Init the dataset and the transform
        :param dataset: RuntimeDataset, the base dataset
        :param transform: ImageCompose, the transform to perform
        """
        self.dataset = dataset
        self.transform = transform

    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run transform
        :param args: None
        :param kwargs: None
        :return: images and the labels
        """
        data_list = []
        label_list = []
        bar = tqdm(range(len(self.dataset)))
        bar.set_description("Generating the dataset from base")
        for image, label in self.dataset:
            if "use_images" in kwargs.keys() and kwargs["use_images"] == 2:
                extra_index = random.randint(0, len(self.dataset)-1)
                extra_image = self.dataset[extra_index][0]
                extra_label = self.dataset[extra_index][1]

                image, label = self.transform(image, label, extra_image, extra_label)
            else:
                image, label = self.transform(image, label)
            data_list.append(image)
            label_list.append(label)
            bar.update(1)
        bar.close()
        data_list = np.array(data_list)
        label_list = np.array(label_list)
        return data_list, label_list

    @staticmethod
    def to_one_hot(label: int) -> np.ndarray:
        """
        Change the int to one-hot
        :param label: int, the label
        :return: np.ndarray, the one-hot label
        """
        t = np.zeros(shape=(10,), dtype=np.float)
        t[label] = 1.0
        return t
