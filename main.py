import time
import argparse
import torch
import VGPAE_classifier
import classifier
import VAE
from utils import get_dataloader, train, train_VAE
from torch import nn

parser = argparse.ArgumentParser(description='VGPAE Experiment')
parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for training (default: 1000)')
parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='number of epochs to train (default: 3000)')
parser.add_argument('--dataset', type=str, default='MNIST',
                    help='dataset (default: letter A)')
parser.add_argument('--actif', type=str, default='lrelu', metavar='Activation',
                    help='activation function (default: LeakyRelu)')
parser.add_argument('--latent_dim1', type=int, default=10, metavar='N',
                    help='dimension of z1 space (default: 2)')
parser.add_argument('--latent_dim2', type=int, default=2, metavar='N',
                    help='dimension of z2 space (default: 2)')
parser.add_argument('--weight', type=float, default=1, metavar='N',
                    help='weight of regularization (default: 1)')
parser.add_argument('--kl', action='store_true', default=False,
                    help='KL loss')
parser.add_argument('--annealing', action='store_true', default=False,
                    help='KL annealing')
parser.add_argument('--fixgp', action='store_true', default=False,
                    help='fix gp parameters')
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
    train_loader, ds = get_dataloader(args.dataset, args.batch_size)
    if args.dataset == 'MNIST':
        img_size = 28
    else:
        img_size = 64
    # initialize model
    in_channel = 1
    enc = classifier.ModelCNN()
    enc.load_state_dict(torch.load('best_model_cnn.pt'))
    enc.eval()
    model = VGPAE_classifier.VGPAE_v2(in_channel, args.latent_dim1, args.latent_dim2, activFun, enc, img_size=img_size,init_para=[0.5,5,15])
    # model = VAE.VAE(in_channel, args.latent_dim1, args.latent_dim2, activFun, img_size=img_size)

    # modify name
    file_name = args.dataset + '_' + model.__class__.__name__ + '_' + \
                str(args.latent_dim1) + '_' + str(args.latent_dim2)

    model = model.to(device)
    # if args.kl:
    #     loss_fun = model.loss_function_kl
    # else:
    loss_fun = model.loss_function

    start_time = time.time()
    train(model, train_loader, args.epochs, device, file_name, args.seed, args.weight, loss_fun, args.annealing, args.fixgp)
    # train_VAE(model, train_loader, args.epochs, device, file_name, args.seed, args.weight, loss_fun, args.annealing)

    print('training time elapsed {}s'.format(time.time() - start_time))
