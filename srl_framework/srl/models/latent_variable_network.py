# Implementation based on https://github.com/ku2482/slac.pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions import Normal
import math

from srl_framework.srl.models.base import BaseModelSRL
from srl_framework.utils.encoder import CnnEncoder, ResNetEncoder, PixelEncoder
from srl_framework.utils.decoder import CnnDecoder, ResNetDecoder, PixelDecoder
from srl_framework.utils.networks import make_mlp

from srl_framework.srl.losses.losses import (
    kl_divergence_seq,
    log_liklihood,
    log_liklihood_from_dist,
    reconstruction_loss,
    calculate_kl_divergence,
)


class FixedGaussian(nn.Module):
    def __init__(self, output_dim, std=1.0):
        super(FixedGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        mu = torch.zeros((x.size(0), self.output_dim)).to(x.device)
        std = torch.ones((x.size(0), self.output_dim)).to(x.device) * self.std
        return mu, std


class Gaussian(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Gaussian, self).__init__()
        self.fixed_std = kwargs["fixed_std"]
        output_dim = output_dim if self.fixed_std else 2 * output_dim
        self.mlp, _ = make_mlp(input_dim=input_dim, output_dim=output_dim)

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.cat(x, dim=-1)
        x = self.mlp(x)
        if self.fixed_std:
            mu = x
            std = torch.ones_like(mu) * self.fixed_std
        else:
            mu, std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(std) + 1e-5
        return mu, std


def build_mlp(
    input_dim,
    output_dim,
    hidden_units=[64, 64],
    hidden_activation=nn.Tanh(),
    output_activation=None,
):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def calculate_gaussian_log_prob(log_std, noise):
    return (-0.5 * noise.pow(2) - log_std).sum(dim=-1, keepdim=True) - 0.5 * math.log(
        2 * math.pi
    ) * log_std.size(-1)


def calculate_log_pi(log_std, noise, action):
    gaussian_log_prob = calculate_gaussian_log_prob(log_std, noise)
    return gaussian_log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(
        dim=-1, keepdim=True
    )


def reparameterize(mean, log_std):
    noise = torch.randn_like(mean)
    action = torch.tanh(mean + noise * log_std.exp())
    return action, calculate_log_pi(log_std, noise, action)


def calculate_kl_divergence(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow_(2)
    t1 = ((p_mean - q_mean) / q_std).pow_(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


def initialize_weight(m):
    if (
        isinstance(m, nn.Linear)
        or isinstance(m, nn.Conv2d)
        or isinstance(m, nn.ConvTranspose2d)
    ):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class LatentNetwork(nn.Module):
    def __init__(
        self,
        img_channels,
        state_dim,
        act_dim,
        seq_len=8,
        latent1_dim=32,
        normalized_obs=False,
        latent2_dim=256,
        img_size=84,
        encoder_args={},
        gaussian_args={},
        lr=0.001,
    ):
        super(LatentNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent1_dim = latent1_dim
        self.latent2_dim = latent2_dim
        self.seq_len = seq_len
        self.std = 1.0

        # p(z1(0)) = N(0, I)
        self.latent1_init_prior = FixedGaussian(latent1_dim, 1.0)
        # p(z2(0) | z1(0))
        self.latent2_init_prior = Gaussian(latent1_dim, latent2_dim, **gaussian_args)
        # p(z1(t+1) | z2(t), a(t))
        self.latent1_prior = Gaussian(
            latent2_dim + act_dim, latent1_dim, **gaussian_args
        )
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_prior = Gaussian(
            latent1_dim + latent2_dim + act_dim, latent2_dim, **gaussian_args
        )

        # q(z1(0) | feat(0))
        self.latent1_init_posterior = Gaussian(state_dim, latent1_dim, **gaussian_args)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.latent2_init_posterior = self.latent2_init_prior
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.latent1_posterior = Gaussian(
            state_dim + latent2_dim + act_dim, latent1_dim, **gaussian_args
        )
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_posterior = self.latent2_prior

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward_predictor = Gaussian(
            2 * latent1_dim + 2 * latent2_dim + act_dim, 1, **gaussian_args
        )

        # feat(t) = x(t) : This encoding is performed deterministically.
        # should be p(x(t) | z1(t), z2(t)). Here x_tilde(t) = dec(z1(t),z2(t)) => p(x(t)|z1(t),z2(t)) = Normal(mu=x_tilde, std =sqrt(0.1))
        if encoder_args["architecture"] == "impala":
            self.encoder = ResNetEncoder(
                img_channels=img_channels,
                feature_dim=state_dim,
                img_size=img_size,
                normalized_obs=normalized_obs,
                squash_latent=encoder_args["squash_latent"],
                normalize=encoder_args["normalize"],
                conv_init=encoder_args["conv_init"],
                linear_init=encoder_args["linear_init"],
            )

            self.decoder = ResNetDecoder(
                feature_dim=latent1_dim + latent2_dim,
                in_dim=self.encoder.output_dim,
                out_dim=img_channels,
                img_size=img_size,
            )
        elif encoder_args["architecture"] == "standard":
            self.encoder = PixelEncoder((img_channels, img_size, img_size), state_dim)
            self.decoder = PixelDecoder(
                (img_channels, img_size, img_size), latent1_dim + latent2_dim
            )
        else:
            self.encoder = CnnEncoder(
                img_channels=img_channels,
                feature_dim=state_dim,
                img_size=img_size,
                architecture=encoder_args["architecture"],
                normalized_obs=normalized_obs,
                squash_latent=encoder_args["squash_latent"],
                normalize=encoder_args["normalize"],
                conv_init=encoder_args["conv_init"],
                linear_init=encoder_args["linear_init"],
                activation=encoder_args["activation"],
                batchnorm=encoder_args["batchnorm"],
                pool=encoder_args["pool"],
                dropout=encoder_args["dropout"],
            )

            self.decoder = CnnDecoder(
                feature_dim=latent1_dim + latent2_dim,
                in_dim=self.encoder.output_dim,
                out_dim=img_channels,
                architecture=encoder_args["architecture"],
                conv_init=encoder_args["conv_init"],
                linear_init=encoder_args["linear_init"],
                activation=encoder_args["activation"],
                batchnorm=encoder_args["batchnorm"],
                pool=encoder_args["pool"],
                dropout=encoder_args["dropout"],
            )

        self.to(self.device)
        print(self.parameters())
        self.apply(initialize_weight)

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def sample_prior(self, actions_seq):
        """ Sample from prior dynamics (with conditionning on the initial frames).
        Inputs:
        ------
            actions_seq   : (N, S, *action_shape) tensor of action sequences.
        Returns:
        ------
            latent1_mean : (N, S+1, L1) prior means of first latent distributions.
            latent1_std : (N, S+1, L1) prior stds of first latent distributions.
        """
        latent1_mean = []
        latent1_std = []

        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = self.latent1_init_prior(actions_seq[:, 0])
        z1 = z1_mean + torch.rand_like(z1_std) * z1_std
        # p(z2(0) | z1(0))
        z2_mean, z2_std = self.latent2_init_prior(z1)
        z2 = z2_mean + torch.rand_like(z2_std) * z2_std

        latent1_mean.append(z1_mean)
        latent1_std.append(z1_std)

        for t in range(1, actions_seq.size(1) + 1):
            # p(z1(t) | z2(t-1), a(t-1))
            z1_mean, z1_std = self.latent1_prior(
                torch.cat([z2, actions_seq[:, t - 1]], dim=1)
            )
            z1 = z1_mean + torch.rand_like(z1_std) * z1_std
            # p(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.latent2_prior(
                torch.cat([z1, z2, actions_seq[:, t - 1]], dim=1)
            )
            z2 = z2_mean + torch.rand_like(z2_std) * z2_std

            latent1_mean.append(z1_mean)
            latent1_std.append(z1_std)

        latent1_mean = torch.stack(latent1_mean, dim=1)
        latent1_std = torch.stack(latent1_std, dim=1)

        return (latent1_mean, latent1_std)

    def sample_posterior(self, features_seq, actions_seq):
        """ Sample from posterior dynamics.
        Inputs:
        ------
            features_seq : (N, S+1, state_dim) tensor of feature sequences.
            actions_seq  : (N, S, *action_space) tensor of action sequences.
        Return:
        ------
            latent1_mean : (N, S+1, L1) posterior means of first latent distributions.
            latent1_std : (N, S+1, L1) posterior stds of first latent distributions.
            latent1_samples : (N, S+1, L1) posterior samples of first latent distributions.
            latent2_samples : (N, S+1, L2) posterior samples of second latent distributions.
        """

        latent1_mean = []
        latent1_std = []
        latent1_samples = []
        latent2_samples = []

        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = self.latent1_init_posterior(features_seq[:, 0])
        z1 = z1_mean + torch.rand_like(z1_std) * z1_std
        # p(z2(0) | z1(0))
        z2_mean, z2_std = self.latent2_init_posterior(z1)
        z2 = z2_mean + torch.rand_like(z2_std) * z2_std

        latent1_mean.append(z1_mean)
        latent1_std.append(z1_std)
        latent1_samples.append(z1)
        latent2_samples.append(z2)

        for t in range(1, actions_seq.size(1) + 1):
            # q(z1(t) | feat(t), z2(t-1), a(t-1))
            z1_mean, z1_std = self.latent1_posterior(
                torch.cat([features_seq[:, t], z2, actions_seq[:, t - 1]], dim=1)
            )
            z1 = z1_mean + torch.rand_like(z1_std) * z1_std
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.latent2_posterior(
                torch.cat([z1, z2, actions_seq[:, t - 1]], dim=1)
            )
            z2 = z2_mean + torch.rand_like(z2_std) * z2_std

            latent1_mean.append(z1_mean)
            latent1_std.append(z1_std)
            latent1_samples.append(z1)
            latent2_samples.append(z2)

        latent1_mean = torch.stack(latent1_mean, dim=1)
        latent1_std = torch.stack(latent1_std, dim=1)
        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        return (latent1_mean, latent1_std, latent1_samples, latent2_samples)

    def calc_latent_loss(self, obs_seq, action_seq, reward_seq, done):
        # Calculate the sequence of features.
        feature_seq = self.encode(obs_seq)
        if action_seq.ndim < 3:
            action_seq = action_seq.unsqueeze(-1)
        if reward_seq.ndim < 3:
            reward_seq = reward_seq.unsqueeze(-1)
        if done.ndim < 3:
            done = done.unsqueeze(-1)

        # Sample from latent variable model.
        (
            latent1_post_mean,
            latent1_post_std,
            latent1_samples,
            latent2_samples,
        ) = self.sample_posterior(feature_seq, action_seq)
        latent1_prior_mean, latent1_prior_std = self.sample_prior(action_seq)

        # Calculate KL divergence loss.
        loss_kld = (
            calculate_kl_divergence(
                latent1_post_mean,
                latent1_post_std,
                latent1_prior_mean,
                latent1_prior_std,
            )
            .mean(dim=0)
            .sum()
        )

        # Prediction loss of images.
        latent_state = torch.cat([latent1_samples, latent2_samples], dim=-1)
        decoded_obs = self.decode(latent_state)
        decoded_obs_seq_std = torch.ones_like(decoded_obs).mul_(self.std)

        obs_noise = (obs_seq - decoded_obs) / (decoded_obs + 1e-8)
        log_likelihood_ = (
            -0.5 * obs_noise.pow(2) - decoded_obs.log()
        ) - 0.5 * math.log(2 * math.pi)
        loss_image = log_likelihood_.mean(dim=0).sum()
        loss_reconstruction = reconstruction_loss(decoded_obs, obs_seq)

        # Prediction loss of rewards.
        x = torch.cat([latent_state[:, :-1], action_seq, latent_state[:, 1:]], dim=-1)
        B, S, X = x.shape
        reward_mean, reward_std = self.reward_predictor(x.view(B * S, X))
        reward_mean = reward_mean.view(B, S, 1)
        reward_std = reward_std.view(B, S, 1)

        reward_noise = (reward_seq - reward_mean) / (reward_std + 1e-8)
        log_likelihood_reward = (
            -0.5 * reward_noise.pow(2) - reward_std.log()
        ) - 0.5 * math.log(2 * math.pi)
        loss_reward = log_likelihood_reward.mul_(1 - done).mean(dim=0).sum()

        return loss_kld, loss_reconstruction, loss_reward, loss_image, decoded_obs

    def infer(self, state, action, rnn_hidden):
        pass

    def get_state(self, obs, detach=False):
        """
        Input:
        ------
            - obs (torch tensor)
            - grad (bool): Set if gradient is required in latter calculations
        Return:
        ------
            - torch tensor
        """
        features = self.encode(obs, detach=detach)
        return features

    def encode(self, x, detach=False):
        """
        """
        B, S, C, H, W = x.size()
        x = self.encoder(x.view(B * S, C, H, W), detach=detach)
        x = x.view(B, S, -1)
        return x

    def decode(self, x):
        """
        """
        B, S, latent_dim = x.size()
        x = x.view(B * S, -1)
        # x = x.view(B * S, latent_dim, 1, 1)
        x = self.decoder(x)
        _, C, W, H = x.size()
        x = x.view(B, S, C, W, H)
        return x
