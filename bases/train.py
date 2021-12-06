from __future__ import print_function

import os
import random
import shutil
from argparse import Namespace
from typing import Dict

import torchvision.transforms
from tqdm import tqdm
import glob
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
from functional.datasets.mixup_dataset import MixupDataset
from util.logger.logger import GlobalLogger
from bases.utils import load_model, AverageMeter, accuracy

# Use CUDA
from util.logger.tensorboards import GlobalTensorboard
from util.tools.draw_util import ImageDrawer

use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

#######################
# Modified by adding transform
global_norm = np.array([[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]])  # (mean & std)
denormalized = torchvision.transforms.Normalize(mean=-global_norm[0] / global_norm[1], std=1 / global_norm[1])
global_label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# End modified
#######################


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform, path):
        #######################
        # Modified by adding "allow_pickle" options
        if os.path.exists(os.path.join(path, 'data.npy')) and os.path.exists(os.path.join(path, 'label.npy')):
            try:
                images = np.load(os.path.join(path, 'data.npy'))
                labels = np.load(os.path.join(path, 'label.npy'))
            except ValueError:
                GlobalLogger().get_logger().warning("Load dataset with numpy failed, load with pickle......")
                try:
                    images = np.load(os.path.join(path, 'data.npy'), allow_pickle=True)
                    labels = np.load(os.path.join(path, 'label.npy'), allow_pickle=True)
                    GlobalLogger().get_logger().info("Load dataset with pickle successful.")
                except ValueError or FileNotFoundError as e:
                    GlobalLogger().get_logger().warning("Unable to load dataset!")
                    raise e
        else:
            GlobalLogger().get_logger().warning("Dataset not found, searching renamed-dataset......")
            image_path = glob.glob(os.path.join(path, "*data*.npy"))
            label_path = glob.glob(os.path.join(path, "*label*.npy"))
            if len(image_path) != len(label_path) or len(image_path) != 1:
                GlobalLogger().get_logger().warning("Multi-implementation of data.npy and label.npy!")
                raise ValueError
            try:
                GlobalLogger().get_logger().warning("Load dataset with renaming dataset......")
                images = np.load(image_path[0], allow_pickle=False)
                labels = np.load(label_path[0], allow_pickle=False)
                GlobalLogger().get_logger().info("Load renamed-dataset with pickle successful.")
            except ValueError as e:
                GlobalLogger().get_logger().warning("Load renamed-dataset with numpy failed, load with pickle......")
                try:
                    images = np.load(image_path[0], allow_pickle=True)
                    labels = np.load(label_path[0], allow_pickle=True)
                    GlobalLogger().get_logger().info("Load renamed-dataset with pickle successful.")
                except ValueError or FileNotFoundError as e:
                    GlobalLogger().get_logger().warning("Unable to load renamed-dataset!")
                    raise e
        # End modified
        #######################
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
        #######################
        # Modified by adding None options
        if self.transform is not None:
            image = self.transform(image)
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            image = self.transform(image)
        # End modified
        #######################
        return image, label

    def __len__(self):
        return len(self.labels)


def _cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)


def train(opt: Namespace, identifier: str):
    #######################
    # Modified by adding the two paths
    dataset_path = opt.data_train
    dataset_eval = opt.data_eval
    checkpoint_path = os.path.join(opt.output_checkpoint_dir, opt.log_name, identifier)
    best_acc_list = dict()
    # End modify
    #######################
    for arch in ['resnet50', 'densenet121']:
        GlobalLogger().get_logger().info("Running model {}".format(arch))
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
        #######################
        # Modified source dataset
        trainset = MyDataset(transform=transform_train, path=dataset_path)
        # End modify
        #######################
        trainloader = data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=4)

        #######################
        # Modified by adding the two paths
        evalloader = None
        if dataset_eval is not None and dataset_eval != "":
            evalset = MyDataset(transform=None, path=dataset_eval)
            evalloader = data.DataLoader(evalset, batch_size=args['batch_size'], shuffle=False, num_workers=4)
        best_eval = 0
        # End modify
        #######################
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
            train_loss, train_acc = _train(trainloader, model, optimizer, enable_tensorboard=opt.enable_tensorboard,
                                           epoch=epoch)
            GlobalLogger().get_logger().debug("Epoch {} with acc in training: {:.2f}".format(epoch + 1, train_acc))

            #######################
            # Modified by adding the two paths
            if opt.enable_tensorboard:
                GlobalTensorboard().get_writer().add_scalar("{}/train/train_loss".format(arch), train_loss,
                                                            (epoch + 1) * len(trainloader))
                GlobalTensorboard().get_writer().add_scalar("{}/train/train_acc".format(arch), train_acc / 100,
                                                            (epoch + 1) * len(trainloader))

            if opt.eval_per_epoch > 0 and evalloader is not None:
                if (epoch + 1) % opt.eval_per_epoch == 0:
                    eval_loss, eval_acc = _eval(evalloader, model)
                    GlobalLogger().get_logger().info(
                        "Epoch {} with acc in evaluation: {:.2f}".format(epoch + 1, eval_acc))
                    GlobalLogger().get_logger().info(
                        "Epoch {} with acc in training: {:.2f}".format(epoch + 1, train_acc))
                    if opt.enable_tensorboard:
                        GlobalTensorboard().get_writer().add_scalar("{}/eval/eval_loss".format(arch), eval_loss,
                                                                    (epoch + 1) * len(evalloader))
                        GlobalTensorboard().get_writer().add_scalar("{}/eval/eval_acc".format(arch), eval_acc / 100,
                                                                    (epoch + 1) * len(evalloader))
                    best_eval = max(eval_acc, best_eval)
            # End modify
            #######################

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
        if best_eval > 0:
            GlobalLogger().get_logger().info("Best acc in evaluation: {:.2f}".format(best_eval))
        best_acc_list[arch] = {"train": best_acc, "test": best_eval} if best_eval > 0 else {"train": best_acc}
    return best_acc_list


def _train(trainloader, model, optimizer, **kwargs):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()  # Original here
    # switch to train mode
    model.train()

    for index, (inputs, soft_labels) in enumerate(trainloader):
        if "enable_tensorboard" in kwargs.keys() and kwargs[
            "enable_tensorboard"] == True and "epoch" in kwargs.keys() and (index + 1) % 100 == 0:
            image_drawer = ImageDrawer(figsize=(32, 16))
            inputs = denormalized(inputs)
            title = []
            for i in soft_labels.numpy():
                classes = ""
                for j in np.where(i > 0)[0]:
                    classes += "{}:{:.2f}\n".format(global_label[j], i[j])
                title.append(classes)
            image_drawer.draw_same_batch(inputs, row=8, title=title)

            image = image_drawer.get_image()
            GlobalTensorboard().get_writer().add_figure("train", image, global_step=index + 1 + len(trainloader) *
                                                                                    kwargs["epoch"])
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


#######################
# Modified by adding the following functions
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


def _eval(testdataloader, model, **kwargs):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    for index, (inputs, soft_labels) in enumerate(testdataloader):
        if "enable_tensorboard" in kwargs.keys() and kwargs[
            "enable_tensorboard"] == True and "epoch" in kwargs.keys() and (index + 1) % 100 == 0:
            image_drawer = ImageDrawer(figsize=(32, 16))
            inputs = denormalized(inputs)
            title = []
            for i in soft_labels.numpy():
                classes = ""
                for j in np.where(i > 0)[0]:
                    classes += "{}:{:.2f}\n".format(global_label[j], i[j])
                title.append(classes)
            image_drawer.draw_same_batch(inputs, row=8, title=title)
            image = image_drawer.get_image()
            GlobalTensorboard().get_writer().add_figure("eval", image, global_step=index + 1 + len(testdataloader) *
                                                                                   kwargs["epoch"])
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = _cross_entropy(outputs, soft_labels)
        acc = accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))
    return losses.avg, accs.avg


# End Modify
#######################

def _save_checkpoint(state, arch, path):
    filepath = os.path.join(path, arch + '.pth.tar')
    torch.save(state, filepath)
