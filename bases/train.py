from __future__ import print_function

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

# Use CUDA
use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class MyDataset(torch.utils.data.Dataset):
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

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)


def _cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)


def train(opt: Namespace, identifier: str):
    dataset_path = opt.data_train
    checkpoint_path = os.path.join(opt.output_checkpoint_dir, opt.log_name, identifier)
    for arch in ['resnet50', 'densenet121']:
        if arch == 'resnet50':
            args = args_resnet
        else:
            args = args_densenet
        assert args['epochs'] <= 200
        # Data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = MyDataset(transform=transform_train, path=dataset_path)
        trainloader = data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
        # Model

        model = load_model(arch)
        best_acc = 0  # best test accuracy

        optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
                                                           **args['optimizer_hyperparameters'])
        if args['scheduler_name'] is not None:
            scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
                                                                                  **args['scheduler_hyperparameters'])
        model = model.cuda()

        GlobalLogger().get_logger().debug("Using config: {}".format(args).replace("\n", " "))
        # Train
        for epoch in tqdm(range(args['epochs'])):
            train_loss, train_acc = _train(trainloader, model, optimizer)
            GlobalLogger().get_logger().debug("Epoch {} with acc in training: {:.2f}".format(epoch + 1, train_acc))

            # save model
            best_acc = max(train_acc, best_acc)
            _save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': train_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, arch=arch, path=checkpoint_path)

            if args['scheduler_name'] is not None:
                scheduler.step()

        GlobalLogger().get_logger().info("Best acc in training: {:.2f}".format(best_acc))
        return best_acc


def _train(trainloader, model, optimizer):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    # switch to train mode
    model.train()

    for (inputs, soft_labels) in trainloader:
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = _cross_entropy(outputs, soft_labels)
        acc = accuracy(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))
    return losses.avg, accs.avg


def evaluate(dataset_path: str, weight_path: str):
    for arch in ['resnet50', 'densenet121']:
        if arch == 'resnet50':
            args = args_resnet
        else:
            args = args_densenet
        # Data
        testset = MyDataset(transform=None, path=dataset_path)
        testdataloader = data.DataLoader(testset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
        # Model

        model = load_model(arch)
        model.load_state_dict(torch.load(weight_path))

        model = model.cuda()

        GlobalLogger().get_logger().debug("Using config: {}".format(args).replace("\n", " "))
        # Train
        test_loss, test_acc = _eval(testdataloader, model)

        GlobalLogger().get_logger().info("Average acc in test: {:.2f}%".format(test_acc * 100))
        GlobalLogger().get_logger().info("Average loss in test: {:.4f}".format(test_loss))


def _eval(testdataloader, model):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    for (inputs, soft_labels) in testdataloader:
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = _cross_entropy(outputs, soft_labels)
        acc = accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))
    return losses.avg, accs.avg


def _save_checkpoint(state, arch, path):
    filepath = os.path.join(path, arch + '.pth.tar')
    torch.save(state, filepath)
