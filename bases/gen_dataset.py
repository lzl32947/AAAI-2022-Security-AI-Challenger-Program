import numpy as np
import torchvision
import random
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
images = []
soft_labels = []
for image, label in dataset:
    image = np.array(image)
    images.append(image)
    soft_label = np.zeros(10)
    soft_label[label] += random.uniform(0, 10)  # an unnormalized soft label vector
    soft_labels.append(soft_label)
images = np.array(images)
soft_labels = np.array(soft_labels)
print(images.shape, images.dtype, soft_labels.shape, soft_labels.dtype)
np.save('../dataset/cifar_10_train/data.npy', images)
np.save('../dataset/cifar_10_train/label.npy', soft_labels)
