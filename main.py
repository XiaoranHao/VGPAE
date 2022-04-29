import time
import argparse
import torch
import VGPAE_model
import classifier
from train import train
from utils import get_loaders
from torch import nn

parser = argparse.ArgumentParser(description='VGPAE Experiment')
parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 1024)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--dataset', type=str, default='MNIST',
                    help='dataset (default: MNIST)')
parser.add_argument('--data', type=str, default='./',
                    help='data root (default: ./)')
parser.add_argument('--save', type=str, default='exp1',
                    help='save file name (default: exp1)')
parser.add_argument('--actif', type=str, default='lrelu', metavar='Activation',
                    help='activation function (default: LeakyRelu)')
parser.add_argument('--latent_dim1', type=int, default=10, metavar='N',
                    help='dimension of z1 space (default: 10)')
parser.add_argument('--latent_dim2', type=int, default=2, metavar='N',
                    help='dimension of z2 space (default: 2)')
parser.add_argument('--weight', type=float, default=1.0, metavar='N',
                    help='weight of regularization (default: 1.0)')
parser.add_argument('--learning_rate', type=float, default=1e-2,
                    help='init learning rate')
parser.add_argument('--warmup_epochs', type=int, default=5,
                    help='num of training epochs in which lr is warmed up')
parser.add_argument('--annealing', action='store_true', default=False,
                    help='KL annealing')
parser.add_argument('--warmup_portion', type=float, default=0.3,
                    help='portion of training epochs in which kl is warmed up')
parser.add_argument('--inducing', type=int, default=500, metavar='N',
                    help='number of inducing points (default: 500)')
parser.add_argument('--fixgp', action='store_true', default=False,
                    help='fix gp parameters')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--double', action='store_true', default=False,
                    help='torch datatype')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)

activations_list = {
    'softplus': nn.Softplus(),
    'lrelu': nn.LeakyReLU(),
    'relu': nn.ReLU()
}
activFun = activations_list[args.actif]

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_loader, test_loader = get_loaders(args)
    args.num_total_iter = len(train_loader) * args.epochs

    if args.dataset == 'MNIST':
        img_size = 28
    else:
        img_size = 64
    # initialize model
    in_channel = 1
    enc = classifier.ModelCNN()
    enc.load_state_dict(torch.load('best_model_10cnn.pt'))
    enc.eval()
    # model = VGPAE_model.VGPAE_v3(in_channel, args.latent_dim1, args.latent_dim2, activFun,
    #                              enc, n_ip=args.inducing, img_size=img_size)
    # model = VGPAE_model.VAE(in_channel, args.latent_dim1, args.latent_dim2, activFun,
    #                         enc, img_size=img_size)
    model = VGPAE_model.Decoder(in_channel, args.latent_dim1, activFun,
                                enc, img_size=img_size)
    model = model.to(device)

    start_time = time.time()
    train(model, train_loader, args, device, log_interval=1)
    print('training time elapsed {}s'.format(time.time() - start_time))
    print('latent_dim: {}, seed: {}'.format(args.latent_dim2, args.seed))
