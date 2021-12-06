import ssl

# Disable ssl check
import torchvision

ssl._create_default_https_context = ssl._create_unverified_context


def cifar10_train():
    return torchvision.datasets.CIFAR10(root="temp", train=True, download=True)


def cifar10_test():
    return torchvision.datasets.CIFAR10(root="temp", train=False, download=True)