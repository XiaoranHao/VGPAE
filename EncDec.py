import torch
from base import *
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, MultivariateNormal, kl, kl_divergence, Independent
import gpytorch
from gpytorch.kernels import ScaleKernel
import math


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim1,
                 latent_dim2, activation, img_size=64,
                 mlp_h_dim=300, n_channels=None):
        super(Encoder, self).__init__()
        if n_channels is None:
            n_channels = [64, 128, 256, 512]
        self.in_channels = in_channels
        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2
        self.activation = activation
        self.img_size = img_size
        self.mlp_h_dim = mlp_h_dim
        self.n_channels = n_channels

    @staticmethod
    def reparameterize(mu, logvar, device):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param device: current device CPU/GPU
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=device)
        return eps * std + mu

    @abc.abstractmethod
    def forward(self, *input):
        pass


class BUCovEncoder(Encoder):
    def __init__(self, layer=conv_block, *args):
        super(BUCovEncoder, self).__init__(*args)
        self.last_layer_size = math.ceil(self.img_size / (2 ** len(self.n_channels))) ** 2
        self.last_layer_width = math.ceil(self.img_size / (2 ** len(self.n_channels)))
        modules = []
        in_channels = self.in_channels
        # Build Encoder
        for channels in self.n_channels:
            modules.append(layer(in_channels, channels, self.activation,
                                 kernel_size=3, stride=2, padding=1))
            in_channels = channels
        self.encoder = nn.Sequential(*modules)

        self.z2_mu = nn.Linear(self.n_channels[-1] * self.last_layer_size, self.latent_dim2)
        self.z2_var = nn.Linear(self.n_channels[-1] * self.last_layer_size, self.latent_dim2)

        self.z2_to_z1 = nn.Sequential(
            fc_block(self.latent_dim2, self.mlp_h_dim, self.activation),
            fc_block(self.mlp_h_dim, self.mlp_h_dim, self.activation)
        )

        self.z1_mu = nn.Linear(self.mlp_h_dim, self.latent_dim1)
        self.z1_var = nn.Linear(self.mlp_h_dim, self.latent_dim1)

    def forward(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        z2_mu = self.z2_mu(result)
        log_z2_var = self.z2_var(result)
        z2 = self.reparameterize(z2_mu, log_z2_var, input.device)

        result = self.z2_to_z1(z2)
        z1_mu = self.z1_mu(result)
        log_z1_var = self.z1_var(result)
        z1 = self.reparameterize(z1_mu, log_z1_var, input.device)

        return [z2, z2_mu, log_z2_var, z1, z1_mu, log_z1_var]


class CovDecoder(Encoder):
    def __init__(self, layer=conv_block, *args):
        super(CovDecoder, self).__init__(*args)
        self.last_layer_size = math.ceil(self.img_size / (2 ** len(self.n_channels))) ** 2
        self.last_layer_width = math.ceil(self.img_size / (2 ** len(self.n_channels)))
        modules = []
        in_channels = self.in_channels
        # Build Decoder
        self.decoder_input = nn.Linear(self.latent_dim2, self.n_channels[-1] * self.last_layer_size)
        self.n_channels.reverse()

        for i in range(len(self.n_channels) - 1):
            if self.img_size == 28 and i == 1:
                modules.append(layer(self.n_channels[i], self.n_channels[i + 1], self.activation,
                                     kernel_size=3, stride=2, padding=1, output_padding=0))
            else:
                modules.append(layer(self.n_channels[i], self.n_channels[i + 1], self.activation,
                                     kernel_size=3, stride=2, padding=1, output_padding=1))

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.n_channels[-1],
                               self.n_channels[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(self.n_channels[-1]),
            self.activation,
            nn.Conv2d(self.n_channels[-1], out_channels=self.in_channels,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def forward(self, input):
        result = self.decoder_input(input)
        size = self.last_layer_width
        result = result.view(-1, self.n_channels[0], size, size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
