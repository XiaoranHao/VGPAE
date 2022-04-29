from torch import nn
import abc


def conv_block(in_channels, out_channels, activation, BN=True, *args, **kwargs):
    if BN:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            activation
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            activation
        )


def convt_block(in_channels, out_channels, activation, BN=True, *args, **kwargs):
    if BN:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            activation
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, *args, **kwargs),
            activation
        )


def fc_block(in_dim, out_dim, activation, BN=True, *args, **kwargs):
    if BN:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim, *args, **kwargs),
            nn.BatchNorm1d(out_dim),
            activation
        )
    else:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim, *args, **kwargs),
            activation
        )


class BaseVAE(nn.Module):

    def __init__(self):
        super(BaseVAE, self).__init__()

    def encode(self, input):
        raise NotImplementedError

    def decode(self, input):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, *input):
        pass

    @abc.abstractmethod
    def loss_function(self, *input):
        pass


