from abc import abstractmethod
from imgaug import augmenters as iaa

from functional.generator_function.global_definition import ImageTransforms


class IAAImageTransform(ImageTransforms):
    def __init__(self, threshold: float):
        super().__init__(threshold)
        self.transform = None

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __call__(self, image, label):
        pass

    def basic_transform(self, image, label):
        if self.check_threshold():
            image = self.transform.augment_image(image)
            return image, label
        else:
            return image, label


class IAAGaussianBlur(IAAImageTransform):
    def __init__(self, threshold: float, **kwargs):
        super().__init__(threshold)
        self.kwargs = kwargs
        self.transform = iaa.GaussianBlur(**self.kwargs)

    def __call__(self, image, label):
        return self.basic_transform(image, label)

    def __str__(self):
        return "IAA.GaussianBlur: {}, {}".format(self.threshold, self.kwargs)
