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
            tuple: (sample, target) where target is class_index of the target class.

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
        dataloader = torch.utils.data.DataLoader(data, batch_size=bs_train, pin_memory=True,
                                                 shuffle=shuffle)
        return dataloader
    else:
        # 1 2 3 ... 26 represent A B C ..Z
        if len(dataset) == 1:
            dataset = '0' + dataset
        # data_root = 'mixedletter/character' + dataset
        data_root = 'Latin/character' + dataset
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


def train(model, train_loader, epochs, device, fn, seed, w_, loss_fun, warm_up=False, fix_param=False,
          future_predict=False, ds=None, log_interval=10):
    N = train_loader.dataset.__len__()
    N_batch = train_loader.batch_size
    if fix_param:
        model.raw_noise.requires_grad = False
        model.covar_module.raw_outputscale.requires_grad = False
        model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        fn = fn + 'fix'
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    if warm_up:
        wu_epoch = 500
        w_ls = torch.ones(epochs) * w_
        w_wu = torch.linspace(0, w_, wu_epoch)
        w_ls[:wu_epoch] = w_wu
        fn = fn + 'wu'
    else:
        w_ls = torch.ones(epochs) * w_

    for epoch in range(1, epochs + 1):
        w = w_ls[epoch - 1]
        if not future_predict:
            for batch_idx, (data, target) in enumerate(train_loader):
                if loss_fun.__name__ == 'loss_function_kl_minibatch':
                    data = data.to(device, dtype=torch.float64)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_fun(data, N, N_batch, *output, M_N=w)
                else:
                    data = data.to(device, dtype=torch.float32)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_fun(data, *output, M_N=w)

                loss[0].backward()
                # for name, param in model.named_parameters():
                #     print(name, torch.isfinite(param.grad).all())
                optimizer.step()
                if epoch % log_interval == 0:
                    if loss_fun.__name__ == 'loss_function':
                        print(
                            f"Epoch: {epoch}, Loss: {loss[0].item()}, Recon_Loss: {loss[1].item()}, entropy_z2: {loss[2].item()},"
                            f"entropy_z1: {loss[3].item()}, log_pz1:{loss[4].item()}, log_pz2z1:{loss[5].item()}")
                    else:
                        print(
                            f"Epoch: {epoch}, Loss: {loss[0].item()}, Recon_Loss: {loss[1].item()}, 'kl_z2': {loss[2].item()},"
                            f"'kl_z1': {loss[3].item()}")
        else:
            time_limit = 50
            time_step = 1
            for batch_idx, (data, target, idx) in enumerate(train_loader):
                data = data.to(device, dtype=torch.float32)
                optimizer.zero_grad()
                output = model(data)
                del data
                # future data
                div, mod_future = find_future(idx, time_step, time_limit)
                idx_target = div * time_limit + mod_future
                target_list = list(idx_target)
                target_img = torch.utils.data.Subset(ds, target_list)
                trainloader_target = torch.utils.data.DataLoader(target_img, batch_size=N_batch, pin_memory=True,
                                                                 shuffle=False)
                target = next(iter(trainloader_target))[0].to(device, dtype=torch.float32)
                loss = loss_fun(target, *output, M_N=w)
                loss[0].backward()

                optimizer.step()
                if epoch % log_interval == 0:
                    if loss_fun.__name__ == 'loss_function':
                        print(
                            f"Epoch: {epoch}, Loss: {loss[0].item()}, Recon_Loss: {loss[1].item()}, entropy_z2: {loss[2].item()},"
                            f"entropy_z1: {loss[3].item()}, log_pz1:{loss[4].item()}, log_pz2z1:{loss[5].item()}")
                    else:
                        print(
                            f"Epoch: {epoch}, Loss: {loss[0].item()}, Recon_Loss: {loss[1].item()}, 'kl_z2': {loss[2].item()},"
                            f"'kl_z1': {loss[3].item()}")

    torch.save(model.state_dict(),
               'models/' + fn + '_w' + str(w_) + '_' + loss_fun.__name__ + '_' + str(epochs) + '_' + str(seed) + '.pth')


def train_VAE(model, train_loader, epochs, device, fn, seed, w_, loss_fun, warm_up=False,
          future_predict=False, ds=None, log_interval=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    if warm_up:
        wu_epoch = 500
        w_ls = torch.ones(epochs) * w_
        w_wu = torch.linspace(0, w_, wu_epoch)
        w_ls[:wu_epoch] = w_wu
        fn = fn + 'wu'
    else:
        w_ls = torch.ones(epochs) * w_

    for epoch in range(1, epochs + 1):
        w = w_ls[epoch - 1]
        if not future_predict:
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device, dtype=torch.float32)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fun(data, *output, M_N=w)

                loss[0].backward()
                # for name, param in model.named_parameters():
                #     print(name, torch.isfinite(param.grad).all())
                optimizer.step()
            if epoch % log_interval == 0:
                print(
                    f"Epoch: {epoch}, Loss: {loss[0].item()}, Recon_Loss: {loss[1].item()}, 'kl_z2': {loss[2].item()},"
                    f"'kl_z1': {loss[3].item()}")
        else:
            time_limit = 50
            time_step = 1
            for batch_idx, (data, target, idx) in enumerate(train_loader):
                data = data.to(device, dtype=torch.float32)
                optimizer.zero_grad()
                output = model(data)
                del data
                # future data
                div, mod_future = find_future(idx, time_step, time_limit)
                idx_target = div * time_limit + mod_future
                target_list = list(idx_target)
                target_img = torch.utils.data.Subset(ds, target_list)
                trainloader_target = torch.utils.data.DataLoader(target_img, batch_size=N_batch, pin_memory=True,
                                                                 shuffle=False)
                target = next(iter(trainloader_target))[0].to(device, dtype=torch.float32)
                loss = loss_fun(target, *output, M_N=w)
                loss[0].backward()

                optimizer.step()
                if epoch % log_interval == 0:
                    print(
                        f"Epoch: {epoch}, Loss: {loss[0].item()}, Recon_Loss: {loss[1].item()}, 'kl_z2': {loss[2].item()},"
                        f"'kl_z1': {loss[3].item()}")

    torch.save(model.state_dict(),
               'models/' + fn + '_w' + str(w_) + '_' + loss_fun.__name__ + '_' + str(epochs) + '_' + str(seed) + '.pth')


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
