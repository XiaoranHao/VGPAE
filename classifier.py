import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import utils

import torch.optim as optim
from torch.utils.data.dataloader import DataLoader


class Config(object):

    def __init__(self):
        self.mode = 'cnn'  # 'cnn' or 'mlp'
        self.torch_seed = 9372

        self.mnist_path = './'
        self.save_path = './best_model_10' + self.mode + '.pt'
        self.num_valid = 10000
        self.batch_size = 2560
        self.eval_batch_size = 1000
        self.num_workers = 0

        self.max_epoch = 1000
        self.max_change = 4
        self.max_patience = 10

        self.initial_lr = 0.001


class ModelCNN(nn.Module):

    def __init__(self):
        super(ModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (5, 5))
        self.conv2 = nn.Conv2d(16, 32, (5, 5))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, input_):
        h1 = F.relu(F.max_pool2d(self.conv1(input_), 2))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(F.max_pool2d(self.conv2(h1), 2))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h2 = h2.view(-1, 512)
        h3 = F.relu(self.fc1(h2))
        h3 = F.dropout(h3, p=0.5, training=self.training)
        h4 = self.fc2(h3)
        return h4


class ModelCNN2(nn.Module):

    def __init__(self):
        super(ModelCNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (5, 5))
        self.conv2 = nn.Conv2d(16, 32, (5, 5))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, input_):
        h1 = F.relu(F.max_pool2d(self.conv1(input_), 2))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(F.max_pool2d(self.conv2(h1), 2))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h2 = h2.view(-1, 512)
        h3 = F.relu(self.fc1(h2))
#         h3 = F.dropout(h3, p=0.5, training=self.training)
#         h4 = self.fc2(h3)
        return h3


def train(model, optimizer, loader):
    model.train()
    loss_sum = 0
    acc_sum = 0
    for idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            loss_sum += loss.item()
            predict = output.max(dim=1)[1]
            acc = predict.eq(target).cpu().sum()
            acc_sum += acc
    return loss_sum / len(loader), acc_sum / len(loader)


def evaluate(model, loader):
    model.eval()
    loss_sum = 0
    acc_sum = 0
    for idx, (data, target) in enumerate(loader):
        with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss_sum += loss.item()
            predict = output.max(dim=1)[1]
            acc = predict.eq(target).cpu().sum()
            acc_sum += acc
    return loss_sum / len(loader), acc_sum / len(loader)


def main(cfg):
    torch.manual_seed(cfg.torch_seed)

    """Prepare data"""
    if cfg.mode == 'cnn':
        data = torchvision.datasets.MNIST('./', train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              utils.Binarize(),
                                              torchvision.transforms.Normalize(
                                                  (0.5,), (0.5,))
                                          ]))
        train_subset, val_subset = torch.utils.data.random_split(
            data, [len(data) - cfg.num_valid, cfg.num_valid], generator=torch.Generator().manual_seed(1))
        train_loader = DataLoader(dataset=train_subset, shuffle=True, pin_memory=True, batch_size=cfg.batch_size)
        val_loader = DataLoader(dataset=val_subset, shuffle=True, pin_memory=True, batch_size=cfg.eval_batch_size)
        data2 = torchvision.datasets.MNIST('./', train=False, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               utils.Binarize(),
                                               torchvision.transforms.Normalize(
                                                   (0.5,), (0.5,))
                                           ]))
        test_loader = DataLoader(dataset=data2, shuffle=False, pin_memory=True, batch_size=cfg.eval_batch_size)

    else:
        raise ValueError('Not supported mode')

    """Set model"""
    if cfg.mode == 'cnn':
        model = ModelCNN()
    else:
        raise ValueError('Not supported mode')

    model.cuda()
    optimizer = optim.Adam(model.parameters(), cfg.initial_lr)

    """Train"""
    best_valid_loss = 1000000
    patience = 0
    change = 0
    status = 'keep_train'

    for epoch in range(cfg.max_epoch):
        print('... Epoch', epoch, status)
        start_time = time.time()
        if status == 'end_train':
            time.sleep(1)
            torch.save(model.state_dict(), cfg.save_path)
            break
        elif status == 'change_lr':
            time.sleep(1)
            model.load_state_dict(torch.load(cfg.save_path))
            lr = cfg.initial_lr * np.power(0.1, change)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        elif status == 'save_param':
            torch.save(model.state_dict(), cfg.save_path)
        else:
            pass

        train_loss, train_acc = train(model, optimizer, train_loader)
        valid_loss, valid_acc = evaluate(model, val_loader)
        print('...... Train loss, accuracy', train_loss, train_acc / cfg.batch_size)
        print('...... Valid loss, best loss, accuracy', valid_loss, best_valid_loss, valid_acc / cfg.eval_batch_size)

        if valid_loss > best_valid_loss:
            patience += 1
            print('......... Current patience', patience)
            if patience >= cfg.max_patience:
                change += 1
                patience = 0
                print('......... Current lr change', change)
                if change >= cfg.max_change:
                    status = 'end_train'  # (load param, stop training)
                else:
                    status = 'change_lr'  # (load param, change learning rate)
            else:
                status = 'keep_train'  # (keep training)
        else:
            best_valid_loss = valid_loss
            patience = 0
            print('......... Current patience', patience)
            status = 'save_param'  # (save param, keep training)

        end_time = time.time()
        print('...... Time:', end_time - start_time)

    test_loss, test_acc = evaluate(model, test_loader)
    print('...... Test loss, accuracy', test_loss, test_acc / cfg.eval_batch_size)


if __name__ == '__main__':

    config = Config()
    main(config)
