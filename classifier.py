import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

import torch.optim as optim
from torch.utils.data.dataloader import DataLoader


class Config(object):

    def __init__(self):

        self.mode = 'cnn'  # 'cnn' or 'mlp'
        self.torch_seed = 9372

        self.mnist_path = './'
        self.save_path = './best_model_' + self.mode + '.pt'
        self.num_valid = 10000
        self.batch_size = 256
        self.eval_batch_size = 1000
        self.num_workers = 0

        self.max_epoch = 1000
        self.max_change = 4
        self.max_patience = 5

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


def train(model, optimizer, loader):
    model.train()
    loss_sum = 0
    acc_sum = 0
    for idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss_sum += loss.data[0]
        loss.backward()
        optimizer.step()

        predict = output.data.max(1)[1]
        acc = predict.eq(target.data).cpu().sum()
        acc_sum += acc
    return loss_sum / len(loader), acc_sum / len(loader)


def evaluate(model, loader):
    model.eval()
    loss_sum = 0
    acc_sum = 0
    for idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss_sum += loss.data[0]

        predict = output.data.max(1)[1]
        acc = predict.eq(target.data).cpu().sum()
        acc_sum += acc
    return loss_sum / len(loader), acc_sum / len(loader)


def main(cfg):
    torch.manual_seed(cfg.torch_seed)

    """Prepare data"""
    if cfg.mode == 'mlp':
        train_data, train_label, valid_data, valid_label, test_data, test_label = parse_mnist(2, cfg.mnist_path, cfg.num_valid, cfg.parse_seed)
    elif cfg.mode == 'cnn':
        train_data, train_label, valid_data, valid_label, test_data, test_label = parse_mnist(4, cfg.mnist_path, cfg.num_valid, cfg.parse_seed)
    else:
        raise ValueError('Not supported mode')

    transform = MNISTTransform()
    train_dataset = MNISTDataset(train_data, train_label, transform=transform)
    valid_dataset = MNISTDataset(valid_data, valid_label, transform=transform)
    test_dataset = MNISTDataset(test_data, test_label, transform=transform)
    train_iter = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    valid_iter = DataLoader(valid_dataset, cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_iter = DataLoader(test_dataset, cfg.eval_batch_size, shuffle=False)

    """Set model"""
    if cfg.mode == 'mlp':
        model = ModelMLP()
    elif cfg.mode == 'cnn':
        model = ModelCNN()
    else:
        raise ValueError('Not supported mode')

    model.cuda(device_id=0)
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

        train_loss, train_acc = train(model, optimizer, train_iter)
        valid_loss, valid_acc = evaluate(model, valid_iter)
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

    test_loss, test_acc = evaluate(model, test_iter)
    print('...... Test loss, accuracy', test_loss, test_acc / cfg.eval_batch_size)


if __name__ == '__main__':
    from config import Config
    config = Config()
    main(config)
