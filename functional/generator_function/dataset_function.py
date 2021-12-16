import ssl

# Disable ssl check
import numpy as np
import torch
import torchvision
from torch import nn

ssl._create_default_https_context = ssl._create_unverified_context


class RuntimeDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        assert len(data) == len(label)
        self.length = len(data)
        self.data = data
        self.label = label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.length


def cifar_10_train():
    label = []
    data = []
    for item in torchvision.datasets.CIFAR10(root="temp", train=True, download=True):
        data.append(np.array(item[0], dtype=np.uint8))
        label.append(item[1])
    one_hot = np.zeros(shape=(len(label),10), dtype=np.float32)
    for index, item in enumerate(label):
        one_hot[index][item] = 1
    data = np.array(data)
    label = np.array(one_hot)
    return RuntimeDataset(data, one_hot)


def cifar_10_test():
    label = []
    data = []
    for item in torchvision.datasets.CIFAR10(root="temp", train=False, download=True):
        data.append(np.array(item[0], dtype=np.uint8))
        label.append(item[1])
    one_hot = np.zeros(shape=(len(label), 10), dtype=np.float32)
    for index, item in enumerate(label):
        one_hot[index][item] = 1
    data = np.array(data)
    return RuntimeDataset(data, one_hot)
