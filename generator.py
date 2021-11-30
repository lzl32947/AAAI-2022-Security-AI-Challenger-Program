import numpy as np

from functional.datasets.mixup_dataset import MixupDataset
import torchvision.transforms as transforms
import torch.utils.data as data


def mix_up_random(dataset_path):
    trainset = MixupDataset(transform=None, path=dataset_path)
    trainloader = data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    data_list = []
    label_list = []
    count = 1
    for d, l in trainloader:
        if count > 50000:
            break
        d = d.numpy().astype(np.uint8)
        l = l.numpy()

        data_list.append(d)
        label_list.append(l)
        count += 1
        if count % 100 == 0:
            print(count)
    data_list = np.array(data_list)
    data_list = np.squeeze(data_list)
    label_list = np.array(label_list)
    label_list = np.squeeze(label_list)
    np.save("output/data.npy", data_list)
    np.save("output/label.npy", label_list)


if __name__ == '__main__':
    mix_up_random("dataset/cifar_10_standard_train")
