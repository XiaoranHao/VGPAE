import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def get_dataloader(dataset, bs_train, shuffle=True):
    if dataset == 'MNIST':
        data = torchvision.datasets.MNIST('./', train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.1307,), (0.3081,))
                                          ]))
        subset = list(range(0, bs_train))
        trainset = torch.utils.data.Subset(data, subset)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=bs_train, pin_memory=True,
                                                 shuffle=shuffle)
        return dataloader
    else:
        # 1 2 3 ... 26 represent A B C ..Z
        if len(dataset) == 1:
            dataset = '0' + dataset

        data_root = 'Latin/character' + dataset
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        torchvision.transforms.Grayscale()
                                        ])

        ds = torchvision.datasets.ImageFolder(root=data_root,
                                              transform=transform)
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(ds, batch_size=bs_train, pin_memory=True,
                                                 shuffle=shuffle)

        return dataloader


def train(model, train_loader, epochs, device, fn, w):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            output = model(data)
            loss = model.loss_function(data, *output, M_N=w)
            loss['loss'].backward()
            optimizer.step()

        if epoch % epochs == 0:
            torch.save(model.state_dict(), 'models/' + fn + '_w100_'
                       + str(epoch) + '.pth')

        print("Epoch: {ep}, Loss: {Loss}, Recon_Loss: {rLoss}, 'entropy_z2': {entropy_z2}, 'entropy_z1': {entropy_z1}, "
              "'log_pz1': {log_p_z1}, 'log_pz2z1': {log_p_z2z1}".format(ep=epoch, Loss=loss['loss'].item(),
                                                                        rLoss=loss['Reconstruction_Loss'].item(),
                                                                        entropy_z2=loss['entropy_z2'].item(),
                                                                        entropy_z1=loss['entropy_z1'].item(),
                                                                        log_p_z1=loss['log_pz1'].item(),
                                                                        log_p_z2z1=loss['log_pz2z1'].item()))


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
            img_new[:, 0, :, :] = img[:, 0, :, :] * std[0] + mean[0]
            img_new[:, 1, :, :] = img[:, 1, :, :] * std[1] + mean[1]
            img_new[:, 2, :, :] = img[:, 2, :, :] * std[2] + mean[2]
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
