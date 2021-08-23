import time
import argparse
import torch
import VGPAE
from utils import get_dataloader, train
from torch import nn

parser = argparse.ArgumentParser(description='VGPAE Experiment')
parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for training (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--dataset', type=str, default='MNIST',
                    help='dataset (default: letter A)')
parser.add_argument('--actif', type=str, default='lrelu', metavar='Activation',
                    help='activation function (default: LeakyRelu)')
parser.add_argument('--latent_dim1', type=int, default=2, metavar='N',
                    help='dimension of z1 space (default: 5)')
parser.add_argument('--latent_dim2', type=int, default=2, metavar='N',
                    help='dimension of z2 space (default: 5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

activations_list = {
    'softplus': nn.Softplus(),
    'lrelu': nn.LeakyReLU(),
    'relu': nn.ReLU()
}
activFun = activations_list[args.actif]
#
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

if __name__ == '__main__':
    train_loader = get_dataloader(args.dataset, args.batch_size)

    # initialize model
    in_channel = 1
    model = VGPAE.VGPAE(in_channel, args.latent_dim1, args.latent_dim2, activFun, img_size=28)

    file_name = args.dataset + '_' + model.__class__.__name__ + '_' + \
                str(args.latent_dim1) + '_' + str(args.latent_dim2)

    model = model.to(device)

    start_time = time.time()
    train(model, train_loader, args.epochs, device, file_name, 0)
    print('training time elapsed {}s'.format(time.time() - start_time))