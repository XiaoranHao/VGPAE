import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class MyDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform):
        super(MyDataset, self).__init__(root, transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target, index) where target is class_index of the target class.

        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index


def find_future(idx, t_step, t_limit=50):
    div, mod = divmod(idx.numpy(), t_limit)
    mod_future = mod + t_step
    mod_future[np.where(mod_future > (t_limit - 1))] = t_limit - 1
    return div, mod_future


def get_dataloader(dataset, bs_train, shuffle=True, future_predict=False):
    if dataset == 'MNIST':
        data = torchvision.datasets.MNIST('./', train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.5,), (0.5,))
                                          ]))
        subset = list(range(0, bs_train))
        trainset = torch.utils.data.Subset(data, subset)
        # trainset = data
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=bs_train, pin_memory=True,
                                                 shuffle=shuffle)
        return dataloader, trainset
    else:
        # 1 2 3 ... 26 represent A B C ..Z
        if len(dataset) == 1:
            dataset = '0' + dataset
        data_root = 'mixedletter/character' + dataset
        # data_root = 'Latin/character' + dataset
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        torchvision.transforms.Grayscale()
                                        ])
        if not future_predict:
            ds = torchvision.datasets.ImageFolder(root=data_root,
                                                  transform=transform)
            # Create the dataloader
            train_loader = torch.utils.data.DataLoader(ds, batch_size=bs_train, pin_memory=True,
                                                       shuffle=shuffle)

        else:
            ds = MyDataset(data_root, transform)
            train_loader = torch.utils.data.DataLoader(ds,
                                                       batch_size=bs_train, pin_memory=True,
                                                       shuffle=shuffle)

        return train_loader, ds


class Binarize(object):
    """ This class introduces a binarization transformation
    """

    def __call__(self, pic):
        return torch.Tensor(pic.size()).bernoulli_(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def _data_transforms_mnist():
    """Get data transforms for mnist."""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        Binarize(),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        Binarize(),

    ])

    return train_transform, valid_transform


def get_data_transforms(dataset):
    if dataset == 'MNIST':
        return _data_transforms_mnist()


def get_loaders(args):
    train_transform, valid_transform = get_data_transforms(args.dataset)
    if args.dataset == 'MNIST':
        train_data = torchvision.datasets.MNIST(root=args.data, train=True, download=True, transform=train_transform)
        test_data = torchvision.datasets.MNIST(root=args.data, train=False, download=True, transform=valid_transform)
    # noinspection PyUnboundLocalVariable
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                              drop_last=True, pin_memory=True)
    # noinspection PyUnboundLocalVariable
    test_queue = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, pin_memory=False)
    return train_queue, test_queue


def backtrans(img, mu_std=None):
    """
    mu_std is Tuple of mean and std : ([0.5],[0.5]) or ([0.5,0.5,0.5],[0.5,0.5,0.5])
    """
    # transform back to [0,1]

    if mu_std is None:
        img_new = 0.5 * img + 0.5
    else:
        mu = mu_std[0]
        std = mu_std[1]
        if len(mu) == 1:
            img_new = mu[0] + std[0] * img
        else:
            img_new = torch.zeros_like(img)
            img_new[:, 0, :, :] = img[:, 0, :, :] * std[0] + mu[0]
            img_new[:, 1, :, :] = img[:, 1, :, :] * std[1] + mu[1]
            img_new[:, 2, :, :] = img[:, 2, :, :] * std[2] + mu[2]
    return img_new


def imshow(img, figsize=(8, 8)):
    """
    figsize is Tuple of height and width :(8,8)
    """
    img_np = img.numpy()
    plt.figure(figsize=figsize)
    #  C x H x W to H x W x C
    plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.show()
