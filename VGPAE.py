import torch
from base import *
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, MultivariateNormal, kl, kl_divergence, Independent
import gpytorch
from gpytorch.kernels import ScaleKernel
import math


class VGPAE(BaseVAE):

    def __init__(self,
                 in_channels,
                 latent_dim1,
                 latent_dim2,
                 activation,
                 img_size=64,
                 mlp_h_dim=100,
                 hidden_dims=None,
                 init_para=None):
        super(VGPAE, self).__init__()

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
        self.z2_mu = nn.Linear(hidden_dims[-1] * self.last_layer_size, latent_dim2)
        self.z2_var = nn.Linear(hidden_dims[-1] * self.last_layer_size, latent_dim2)

        self.z2_to_z1 = nn.Sequential(
            fc_block(latent_dim2, mlp_h_dim, activation),
            fc_block(mlp_h_dim, mlp_h_dim, activation)
        )

        self.z1_mu = nn.Linear(mlp_h_dim, latent_dim1)
        self.z1_var = nn.Linear(mlp_h_dim, latent_dim1)

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
        noise_init = init_para[0] + torch.randn(latent_dim2) * 0.01
        output_init = init_para[1] + torch.randn(latent_dim2) * 0.1
        length_init = init_para[2] + torch.randn(latent_dim1) * 0.1
        self.covar_module.initialize(outputscale=output_init)
        self.noise_initialize(noise_init)
        self.covar_module.base_kernel.initialize(lengthscale=length_init)

    @property
    def noise(self):
        return torch.exp(self.raw_noise)

    def noise_initialize(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        #         self.__getattr__('raw_noise').data.fill_(torch.log(value))
        if torch.min(value) <= 0.0:
            raise RuntimeError("negative noise occurs")
        value = torch.log(value)
        try:
            self.__getattr__('raw_noise').data.copy_(value.expand_as(self.__getattr__('raw_noise')))
        except RuntimeError:
            self.__getattr__('raw_noise').data.copy_(value.view_as(self.__getattr__('raw_noise')))

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
        z2_mu = self.z2_mu(result)
        log_z2_var = self.z2_var(result)

        return [z2_mu, log_z2_var]

    def encode_to_z1(self, input):
        """
        encodes z2 to z1
        :param input:
        :return:
        """
        result = self.z2_to_z1(input)

        z1_mu = self.z1_mu(result)
        log_z1_var = self.z1_var(result)

        return [z1_mu, log_z1_var]

    def decode(self, z2):
        """
        Maps the given latent codes
        onto the image space.
        :param z2: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z2)
        size = self.last_layer_width
        result = result.view(-1, self.last_h_dim, size, size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

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

    def forward(self, input, **kwargs):
        z2_mu, log_z2_var = self.encode(input)
        z2 = self.reparameterize(z2_mu, log_z2_var, input.device)
        z1_mu, log_z1_var = self.encode_to_z1(z2)
        z1 = self.reparameterize(z1_mu, log_z1_var, input.device)

        recon = self.decode(z2)
        return [recon, z2, z2_mu, log_z2_var, z1, z1_mu, log_z1_var]

    def loss_function(self, input, *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1))
        :param input:
        :param args:
        :param kwargs:
        :return:
        """
        input = input
        recons, z2, z2_mu, log_z2_var, z1, z1_mu, log_z1_var = args
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input, reduction='sum')

        q_z1 = Independent(Normal(loc=z1_mu, scale=torch.exp(0.5 * log_z1_var)),
                           reinterpreted_batch_ndims=1)
        entropy_z1 = q_z1.entropy().sum()
        q_z2 = Independent(Normal(loc=z2_mu, scale=torch.exp(0.5 * log_z2_var)),
                           reinterpreted_batch_ndims=1)
        entropy_z2 = q_z2.entropy().sum()

        p_z1 = Independent(Normal(loc=torch.zeros_like(z1_mu), scale=torch.ones_like(log_z1_var)),
                           reinterpreted_batch_ndims=1)
        log_p_z1 = p_z1.log_prob(z1).sum()

        n_x = len(input)
        sig_y = self.noise.clone()
        eye = torch.eye(n_x, device=input.device)
        batch_eye = eye.expand(self.latent_dim2, n_x, n_x).clone()
        noise = sig_y.view(-1, 1, 1) * batch_eye
        kernel = self.covar_module(z1, z1).evaluate() + noise

        p_z2z1 = MultivariateNormal(loc=torch.zeros_like(z2_mu.T), covariance_matrix=kernel)
        log_p_z2z1 = p_z2z1.log_prob(z2.T).sum()

        loss = recons_loss - kld_weight * (entropy_z2 + entropy_z1 + log_p_z1 + log_p_z2z1)
        return loss, recons_loss, entropy_z2, entropy_z1, log_p_z1, log_p_z2z1

    def loss_function_kl(self, input, *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1))
        :param input:
        :param args:
        :param kwargs:
        :return:
        """

        input = input
        recons, z2, z2_mu, log_z2_var, z1, z1_mu, log_z1_var = args
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input, reduction='sum')

        q_z1 = Independent(Normal(loc=z1_mu, scale=torch.exp(0.5 * log_z1_var)),
                           reinterpreted_batch_ndims=1)
        p_z1 = Independent(Normal(loc=torch.zeros_like(z1_mu), scale=torch.ones_like(log_z1_var)),
                           reinterpreted_batch_ndims=1)
        kl_z1 = kl_divergence(q_z1, p_z1).sum()

        q_z2 = MultivariateNormal(loc=z2_mu.T, scale_tril=torch.diag_embed(torch.exp(0.5 * log_z2_var.T)))

        n_x = len(input)
        sig_y = self.noise.clone()
        eye = torch.eye(n_x, device=input.device)
        batch_eye = eye.expand(self.latent_dim2, n_x, n_x).clone()
        noise = sig_y.view(-1, 1, 1) * batch_eye
        kernel = self.covar_module(z1, z1).evaluate() + noise
        p_z2z1 = MultivariateNormal(loc=torch.zeros_like(z2_mu.T), covariance_matrix=kernel)

        kl_z2 = kl_divergence(q_z2, p_z2z1).sum()

        loss = recons_loss + kld_weight * (kl_z1 + kl_z2)
        return loss, recons_loss, kl_z2, kl_z1
