import torch

import os
import random
import shutil
from argparse import Namespace
from typing import Dict

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from configs.train_config import args_resnet, args_densenet
from util.logger.logger import GlobalLogger
from bases.utils import load_model, AverageMeter, accuracy


class MixupDataset(torch.utils.data.Dataset):
    def __init__(self, transform, path):
        images = np.load(os.path.join(path, 'data.npy'))
        labels = np.load(os.path.join(path, 'label.npy'))
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True)  # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
        self.length = len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        other = np.random.choice(np.where(self.labels != label)[0])
        alpha = np.random.random() * 0.4 + 0.3

        if self.transform is not None:
            image = self.transform(image)
            other_image = self.transform(self.images[other])
        else:
            image = np.array(image)
            other_image = np.array(self.images[other])
        image = image * alpha + other_image * (1 - alpha)
        label = label * alpha + self.labels[other] * (1 - alpha)
        return image, label

    def __len__(self):
        return self.length
