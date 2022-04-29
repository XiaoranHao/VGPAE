import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, MultivariateNormal, kl, kl_divergence, Independent
import gpytorch
from gpytorch.kernels import ScaleKernel
import math
import EncDec
from base import *
from torch.distributions.bernoulli import Bernoulli


class VGPAE(nn.Module):
    def __init__(self,
                 in_channels,
                 latent_dim1,
                 latent_dim2,
                 activation,
                 img_size=28,
                 n_channels=None,
                 init_para=None):
        super(VGPAE, self).__init__()

        self.encoder = EncDec.BUCovEncoder(in_channels=in_channels, latent_dim1=latent_dim1,
                                           latent_dim2=latent_dim2, activation=activation,
                                           img_size=img_size, n_channels=n_channels)
        self.decoder = EncDec.CovDecoder(in_channels=in_channels, latent_dim1=latent_dim1,
                                         latent_dim2=latent_dim2, activation=activation,
                                         img_size=img_size, n_channels=n_channels)

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
        noise_init = init_para[0] + torch.randn(latent_dim2) * 0.1
        output_init = init_para[1] + torch.randn(latent_dim2) * 0.1
        length_init = init_para[2] + torch.randn(latent_dim1) * 0.1
        self.covar_module.initialize(outputscale=output_init)
        self.noise_initialize(noise=noise_init)
        self.covar_module.base_kernel.initialize(lengthscale=length_init)

    @property
    def noise(self):
        return torch.exp(self.raw_noise)

    def noise_initialize(self, noise):
        if not torch.is_tensor(noise):
            noise = torch.as_tensor(noise).to(self.raw_noise)
        if torch.min(noise) <= 0.0:
            raise RuntimeError("negative noise occurs")
        noise = torch.log(noise)
        try:
            self.__getattr__('raw_noise').data.copy_(noise.expand_as(self.__getattr__('raw_noise')))
        except RuntimeError:
            self.__getattr__('raw_noise').data.copy_(noise.view_as(self.__getattr__('raw_noise')))

    def loss_function(self, input, *args,
                      **kwargs):
        input = input
        recons, z2, z2_mu, log_z2_var, z1, z1_mu, log_z1_var = args
        kld_weight = kwargs['M_N']

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
        batch_eye = eye.expand(self.encoder.latent_dim2, n_x, n_x).clone()
        noise = sig_y.view(-1, 1, 1) * batch_eye
        kernel = self.covar_module(z1, z1).evaluate() + noise
        p_z2z1 = MultivariateNormal(loc=torch.zeros_like(z2_mu.T), covariance_matrix=kernel)

        kl_z2 = kl_divergence(q_z2, p_z2z1).sum()

        loss = recons_loss + kld_weight * (kl_z1 + kl_z2)
        return loss, recons_loss, kl_z2, kl_z1

    def forward(self, input):
        z2, z2_mu, log_z2_var = self.encoder(input)
        recon = self.decoder(z2)
        return [recon, z2, z2_mu, log_z2_var]


class VGPAE_v2(VGPAE):
    def __init__(self,
                 in_channels,
                 latent_dim1,
                 latent_dim2,
                 activation,
                 z1_encoder,
                 img_size=28,
                 n_channels=None,
                 init_para=None):
        super(VGPAE_v2, self).__init__(in_channels, latent_dim1, latent_dim2, activation,
                                       img_size, n_channels, init_para)
        self.encoder = EncDec.CovEncoder(in_channels=in_channels, latent_dim1=latent_dim1,
                                         latent_dim2=latent_dim2, activation=activation,
                                         img_size=img_size, n_channels=n_channels)
        self.z1 = z1_encoder

    def forward(self, input):
        input = 2 * input - 1.0
        with torch.no_grad():
            z1 = self.z1(input)
        z2, z2_mu, log_z2_var = self.encoder(input)
        # recon = self.decoder(z2)
        # return [recon, z2, z2_mu, log_z2_var, z1]
        logit = self.decoder(z2)
        return [Bernoulli(logits=logit), z2, z2_mu, log_z2_var, z1]

    def loss_function(self, input, *args, **kwargs):
        input = input
        recons, z2, z2_mu, log_z2_var, z1 = args
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input, reduction='sum')
        q_z2 = MultivariateNormal(loc=z2_mu.T, scale_tril=torch.diag_embed(torch.exp(0.5 * log_z2_var.T)))
        n_x = len(input)
        sig_y = self.noise.clone()
        eye = torch.eye(n_x, device=input.device)
        batch_eye = eye.expand(self.encoder.latent_dim2, n_x, n_x).clone()
        noise = sig_y.view(-1, 1, 1) * batch_eye
        kernel = self.covar_module(z1, z1).evaluate() + noise
        p_z2z1 = MultivariateNormal(loc=torch.zeros_like(z2_mu.T), covariance_matrix=kernel)
        kl_z2 = kl_divergence(q_z2, p_z2z1).sum()
        loss = recons_loss + kld_weight * kl_z2
        return loss, recons_loss, kl_z2


class VGPAE_v3(VGPAE_v2):
    def __init__(self,
                 in_channels,
                 latent_dim1,
                 latent_dim2,
                 activation,
                 z1_encoder,
                 n_ip=300,
                 img_size=28,
                 n_channels=None,
                 init_para=None):
        super(VGPAE_v3, self).__init__(in_channels, latent_dim1, latent_dim2, activation, z1_encoder,
                                       img_size, n_channels, init_para)
        # number of inducing points
        self.n_ip = n_ip

        self.register_parameter(name="variational_input", param=torch.nn.Parameter(torch.randn(n_ip, latent_dim1)))
        self.register_parameter(name="variational_mean", param=torch.nn.Parameter(torch.randn(latent_dim2, n_ip)))

        self.register_parameter(name="variational_cov_log_diag",
                                param=torch.nn.Parameter(2 * torch.ones(latent_dim2, n_ip)))
        self.register_parameter(name="variational_cov",
                                param=torch.nn.Parameter(torch.randn(latent_dim2, n_ip, n_ip)))

    @property
    def variational_covar(self):
        matrix_mask = self.variational_cov.tril(-1)
        lm = matrix_mask + torch.diag_embed(self.variational_cov_log_diag.exp())
        # lm = self.variational_cov
        H = lm.bmm(lm.transpose(-1, -2))
        return H

    def loss_function(self, input, *args, **kwargs):
        N, N_batch, recons, z2, z2_mu, log_z2_var, z1 = args
        kld_weight = kwargs['klw']

        # recons_loss = N / N_batch * F.mse_loss(recons, input, reduction='sum')
        recon = recons.log_prob(input)
        recons_loss = N / N_batch * (- torch.sum(recon, dim=[0, 1, 2, 3]))
        kl_z2 = self.gp_KL_ubo(z2_mu, log_z2_var, z1, N, N_batch)

        loss = recons_loss + kld_weight * kl_z2
        # with torch.no_grad():
        #     q_z2 = MultivariateNormal(loc=z2_mu.T, scale_tril=torch.diag_embed(torch.exp(0.5 * log_z2_var.T)))
        #     n_x = len(input)
        #     sig_y = self.noise.clone()
        #     eye = torch.eye(n_x, device=input.device)
        #     batch_eye = eye.expand(self.encoder.latent_dim2, n_x, n_x).clone()
        #     noise = sig_y.view(-1, 1, 1) * batch_eye
        #     kernel = self.covar_module(z1, z1).evaluate() + noise
        #     p_z2z1 = MultivariateNormal(loc=torch.zeros_like(z2_mu.T), covariance_matrix=kernel)

        # kl_z2_real = kl_divergence(q_z2, p_z2z1).sum()
        # return loss, recons_loss, kl_z2
        return loss / N, recons_loss / N, kl_z2 / N

    def gp_KL_ubo(self, *args, **kwargs):
        z2_mu, log_z2_var, z1, N, N_hat = args

        Knn = self.covar_module(z1, z1)
        Knn = Knn.add_jitter(1e-5)
        Knm = self.covar_module(z1, self.variational_input)
        Kmn = Knm.transpose(-1, -2)
        Kmm = self.covar_module(self.variational_input, self.variational_input)
        Kmm = Kmm.add_jitter(1e-5)

        noise = self.noise.clone()
        diff = (Kmm.inv_matmul(self.variational_mean.unsqueeze(-1), Knm.evaluate()).squeeze(-1) - z2_mu.T)
        A = noise.pow(-1).dot((diff * diff).sum(1))
        B = noise.pow(-1).dot(log_z2_var.T.exp().sum(1))
        Q = Kmm.inv_matmul(Kmn.evaluate(), Knm.evaluate())
        Q_ = (Knn - Q).diag().sum(1)
        C = noise.pow(-1).dot(Q_)
        tr = (Kmm.inv_matmul(Kmn.matmul(Knm).evaluate(), Kmm.inv_matmul(self.variational_covar))).diagonal(dim1=-1,
                                                                                                           dim2=-2).sum(
            1)
        D = noise.pow(-1).dot(tr)
        E = log_z2_var.sum()

        F = 0.5 * N * noise.log().sum() - 0.5 * N
        # q_ind = MultivariateNormal(loc=self.variational_mean, covariance_matrix=self.variational_covar)
        # p_prior = MultivariateNormal(loc=torch.zeros_like(self.variational_mean), covariance_matrix=Kmm.evaluate())

        # Compute kld_qu_pu
        tr1 = Kmm.inv_matmul(self.variational_covar).diagonal(dim1=-1, dim2=-2).sum()
        qf1, logdetK = Kmm.inv_quad_logdet(inv_quad_rhs=self.variational_mean.unsqueeze(-1), logdet=True,
                                           reduce_inv_quad=True)
        LH = torch.cholesky(self.variational_covar)
        logdetH = 2 * torch.sum(torch.log(torch.diagonal(LH, dim1=-1, dim2=-2)))
        kld_qu_pu = 0.5 * (tr1 + qf1 - self.encoder.latent_dim2 * self.n_ip + logdetK - logdetH).sum()

        # kld_qu_pu = kl_divergence(q_ind, p_prior).sum()

        return 0.5 * N / N_hat * (A + B + C + D - E) + F + kld_qu_pu


class VAE(VGPAE_v2):
    def __init__(self,
                 in_channels,
                 latent_dim1,
                 latent_dim2,
                 activation,
                 z1_encoder,
                 img_size=28,
                 n_channels=None,
                 init_para=None):
        super(VAE, self).__init__(in_channels, latent_dim1, latent_dim2, activation, z1_encoder,
                                  img_size, n_channels, init_para)

        self.z1_to_z2 = nn.Sequential(
            fc_block(latent_dim1, 300, activation),
            fc_block(300, 500, activation))
        self.prior_z2_mu = nn.Linear(500, latent_dim2)
        self.prior_z2_var = nn.Linear(500, latent_dim2)

    def loss_function(self, input, *args, **kwargs):
        _, N, recons, z2, z2_mu, log_z2_var, prior_z2_mu, prior_z2_log_var = args
        kld_weight = kwargs['klw']

        recon = recons.log_prob(input)
        recons_loss = - torch.sum(recon)

        q_z2 = Independent(Normal(loc=z2_mu, scale=torch.exp(0.5 * log_z2_var)),
                           reinterpreted_batch_ndims=1)

        p_z2 = Independent(Normal(loc=prior_z2_mu, scale=torch.exp(0.5 * prior_z2_log_var)),
                           reinterpreted_batch_ndims=1)

        kl_z2 = kl_divergence(q_z2, p_z2).sum()
        loss = recons_loss + kld_weight * kl_z2

        return loss / N, recons_loss / N, kl_z2 / N

    def forward(self, input):
        input = 2 * input - 1.0
        with torch.no_grad():
            z1 = self.z1(input)
        z2, z2_mu, log_z2_var = self.encoder(input)
        inter_z2 = self.z1_to_z2(z1)
        prior_z2_mu = self.prior_z2_mu(inter_z2)
        prior_z2_log_var = self.prior_z2_var(inter_z2)

        logit = self.decoder(z2)
        return [Bernoulli(logits=logit), z2, z2_mu, log_z2_var, prior_z2_mu, prior_z2_log_var]


class Decoder(nn.Module):
    def __init__(self, in_channels, latent_dim1,
                 activation, z1_encoder, img_size=64, n_channels=None):
        super(Decoder, self).__init__()
        if n_channels is None:
            n_channels = [64, 128, 256, 512]

        self.decoder = EncDec.CovDecoder(in_channels=in_channels, latent_dim1=latent_dim1,
                                         latent_dim2=latent_dim1, activation=activation,
                                         img_size=img_size, n_channels=n_channels)
        self.z1 = z1_encoder

    def forward(self, input):
        input = 2 * input - 1.0
        with torch.no_grad():
            z1 = self.z1(input)
        logit = self.decoder(z1)
        return Bernoulli(logits=logit), z1

    def loss_function(self, input, *args, **kwargs):
        _, N, recons, z1 = args
        recon = recons.log_prob(input)
        recons_loss = - torch.sum(recon)
        return recons_loss/N, recons_loss/N, recons_loss/N
