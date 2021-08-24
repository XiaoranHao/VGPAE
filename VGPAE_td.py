import torch
from VGPAE import VGPAE
from base import *
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, MultivariateNormal, kl, kl_divergence, Independent
import gpytorch
from gpytorch.kernels import ScaleKernel
import math


class VGPAEtd(VGPAE):

    def __init__(self,
                 in_channels,
                 latent_dim1,
                 latent_dim2,
                 activation,
                 img_size=64,
                 mlp_h_dim=100,
                 hidden_dims=None,
                 init_para=None):
        super(VGPAEtd, self).__init__(
                 in_channels,
                 latent_dim1,
                 latent_dim2,
                 activation,
                 img_size=img_size,
                 mlp_h_dim=mlp_h_dim,
                 hidden_dims=hidden_dims,
                 init_para=init_para)

        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2
        self.img_size = img_size
        self.in_channels = in_channels
        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
        self.last_layer_size = math.ceil(img_size / (2 ** len(hidden_dims))) ** 2
        self.last_layer_width = math.ceil(img_size / (2 ** len(hidden_dims)))
        self.last_h_dim = hidden_dims[-1]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(conv_block(in_channels, h_dim, activation,
                                      kernel_size=3, stride=2, padding=1)
                           )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.z1_mu = nn.Linear(hidden_dims[-1] * self.last_layer_size, latent_dim1)
        self.z1_var = nn.Linear(hidden_dims[-1] * self.last_layer_size, latent_dim1)

        self.z1_to_z2 = nn.Sequential(
            fc_block(latent_dim1, mlp_h_dim, activation),
            fc_block(mlp_h_dim, mlp_h_dim, activation)
        )

        self.z2_mu = nn.Linear(mlp_h_dim, latent_dim2)
        self.z2_var = nn.Linear(mlp_h_dim, latent_dim2)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim2, hidden_dims[-1] * self.last_layer_size)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            if self.img_size == 28 and i == 1:
                modules.append(convt_block(hidden_dims[i], hidden_dims[i + 1], activation,
                                           kernel_size=3, stride=2, padding=1, output_padding=0))
            else:
                modules.append(convt_block(hidden_dims[i], hidden_dims[i + 1], activation,
                                           kernel_size=3, stride=2, padding=1, output_padding=1))

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            activation,
            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels,
                      kernel_size=3, padding=1),
            nn.Tanh())

        # GP covariance module
        self.covar_module = ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim1,
                                                                   lengthscale_constraint=gpytorch.constraints.Positive(
                                                                       transform=torch.exp, inv_transform=torch.log)),
                                        batch_shape=torch.Size([latent_dim2]))

        # register parameters
        self.register_parameter(name="raw_noise", param=torch.nn.Parameter(torch.zeros(latent_dim2)))
        # initialization
        if init_para is None:
            init_para = [1, 1, 1]
        noise_init = init_para[0] + torch.randn(latent_dim2) * 0.2
        output_init = init_para[1] + torch.randn(latent_dim2) * 0.2
        length_init = init_para[2] + torch.randn(latent_dim1) * 0.2
        self.covar_module.initialize(outputscale=output_init)
        self.noise_initialize(noise_init)
        self.covar_module.base_kernel.initialize(lengthscale=length_init)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        z1_mu = self.z1_mu(result)
        log_z1_var = self.z1_var(result)

        return [z1_mu, log_z1_var]

    def encode_to_z2(self, input):
        """
        encodes z2 to z1
        :param input:
        :return:
        """
        result = self.z1_to_z2(input)

        z2_mu = self.z2_mu(result)
        log_z2_var = self.z2_var(result)

        return [z2_mu, log_z2_var]

    def forward(self, input, **kwargs):
        z1_mu, log_z1_var = self.encode(input)
        z1 = self.reparameterize(z1_mu, log_z1_var, input.device)
        z2_mu, log_z2_var = self.encode_to_z2(z1)
        z2 = self.reparameterize(z2_mu, log_z2_var, input.device)

        recon = self.decode(z2)
        return [recon, z2, z2_mu, log_z2_var, z1, z1_mu, log_z1_var]

