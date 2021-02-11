import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import math
from srl_framework.utils.distributions import Bernoulli, Categorical, Gaussian
from srl_framework.utils.networks import make_mlp
from srl_framework.utils.encoder import CnnEncoder, ResNetEncoder, PixelEncoder


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Policy(nn.Module):
    """
    """

    def __init__(
        self, encoder, **kwargs
    ):  # state_dim, action_dim, act_limit, param, use_cnn=False, img_channels= 3, envtype = "Box"):
        super(Policy, self).__init__()
        self.use_encoder = kwargs["use_cnn"]
        if kwargs["use_cnn"]:
            if encoder:
                self.encoder = encoder
            else:
                encoder_args = kwargs["encoder_args"]
                if encoder_args["architecture"] == "impala":
                    self.encoder = ResNetEncoder(
                        img_channels=kwargs["img_channels"],
                        feature_dim=kwargs["state_dim"],
                        img_size=kwargs["img_size"],
                        normalized_obs=kwargs["normalized_obs"],
                        squash_latent=encoder_args["squash_latent"],
                        normalize=encoder_args["normalize"],
                        conv_init=encoder_args["conv_init"],
                        linear_init=encoder_args["linear_init"],
                    )
                elif encoder_args["architecture"] == "standard":
                    self.encoder = PixelEncoder(
                        (
                            kwargs["img_channels"],
                            kwargs["img_size"],
                            kwargs["img_size"],
                        ),
                        kwargs["feature_dim"],
                    )
                else:
                    self.encoder = CnnEncoder(
                        img_channels=kwargs["img_channels"],
                        feature_dim=kwargs["state_dim"],
                        img_size=kwargs["img_size"],
                        normalized_obs=kwargs["normalized_obs"],
                        architecture=encoder_args["architecture"],
                        squash_latent=encoder_args["squash_latent"],
                        normalize=encoder_args["normalize"],
                        conv_init=encoder_args["conv_init"],
                        linear_init=encoder_args["linear_init"],
                        activation=encoder_args["activation"],
                        batchnorm=encoder_args["batchnorm"],
                        pool=encoder_args["pool"],
                        dropout=encoder_args["dropout"],
                    )

        self.vae = True if kwargs["latent_type"] == "vae" else False

        self.mlp, dist_in_dim = make_mlp(
            input_dim=kwargs["input_dim_actor"],
            output_dim=0,
            architecture=[64, 64],
            activation="ReLU",
            output_layer=False,
            batchnorm=False,
            dropout=False,
            init="orthogonal",
        )
        self.dist = Gaussian(dist_in_dim, kwargs["action_dim"], fixed_std=False)

        if kwargs["envtype"] == "Discrete":
            self.dist = Categorical(dist_in_dim, kwargs["action_dim"])
            self.act_limit = None
        elif kwargs["envtype"] == "Box":
            self.dist = Gaussian(
                num_inputs=dist_in_dim,
                num_outputs=kwargs["action_dim"],
                fixed_std=True,
                log_std_min=-20,
                log_std_max=2,
            )
            self.act_limit = kwargs["action_limit"]
        elif kwargs["envtype"] == "MultiBinary":
            self.dist = Bernoulli(dist_in_dim, kwargs["action_dim"])
            self.act_limit = kwargs["action_limit"]
        else:
            raise NotImplementedError

    def forward(self, state, action=None, detach_encoder=False):
        features = (
            self.encoder(state, detach=detach_encoder) if self.use_encoder else state
        )
        if self.use_encoder and self.vae:
            features, _ = features.chunk(2, dim=-1)  # take mean as input
        features = self.mlp(features)
        pi = self.dist(features)
        if action is not None:
            if action.ndim <= 1:
                action = action.unsqueeze(-1)
            log_prob = pi.log_probs(action)
        else:
            log_prob = None
        return pi, log_prob


class ValueFunction(nn.Module):
    """
    Value Function:
    ----------------------
    Value Function takes state as input and returns a single state value.
    Use in discrete action spaces and algorithms such as DQL.
    """

    def __init__(self, encoder=None, **kwargs):
        super(ValueFunction, self).__init__()
        self.use_encoder = kwargs["use_cnn"]
        if kwargs["use_cnn"]:
            if encoder:
                self.encoder = encoder
            else:
                encoder_args = kwargs["encoder_args"]
                if encoder_args["architecture"] == "impala":
                    self.encoder = ResNetEncoder(
                        img_channels=kwargs["img_channels"],
                        feature_dim=kwargs["feature_dim"],
                        img_size=kwargs["img_size"],
                        normalized_obs=kwargs["normalized_obs"],
                        squash_latent=encoder_args["squash_latent"],
                        normalize=encoder_args["normalize"],
                        conv_init=encoder_args["conv_init"],
                        linear_init=encoder_args["linear_init"],
                    )
                elif encoder_args["architecture"] == "standard":
                    self.encoder = PixelEncoder(
                        (
                            kwargs["img_channels"],
                            kwargs["img_size"],
                            kwargs["img_size"],
                        ),
                        kwargs["feature_dim"],
                    )
                else:
                    self.encoder = CnnEncoder(
                        img_channels=kwargs["img_channels"],
                        feature_dim=kwargs["feature_dim"],
                        img_size=kwargs["img_size"],
                        normalized_obs=kwargs["normalized_obs"],
                        architecture=encoder_args["architecture"],
                        squash_latent=encoder_args["squash_latent"],
                        normalize=encoder_args["normalize"],
                        conv_init=encoder_args["conv_init"],
                        linear_init=encoder_args["linear_init"],
                        activation=encoder_args["activation"],
                        batchnorm=encoder_args["batchnorm"],
                        pool=encoder_args["pool"],
                        dropout=encoder_args["dropout"],
                    )
                kwargs["input_dim_critic"] = kwargs["feature_dim"]
        self.vae = True if kwargs["latent_type"] == "vae" else False

        self.mlp, _ = make_mlp(
            input_dim=kwargs["input_dim_critic"],
            output_dim=1,
            architecture=[64, 64],
            activation="ReLU",
            output_layer=True,
            batchnorm=False,
            dropout=False,
            init="orthogonal",
        )

    def forward(self, state, detach_encoder=False):
        features = (
            self.encoder(state, detach=detach_encoder) if self.use_encoder else state
        )
        if self.use_encoder and self.vae:
            features, _ = features.chunk(2, dim=-1)  # take mean as input
        value = self.mlp(features)
        return torch.squeeze(value, -1)  # squeeze ?


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(
            axis=-1
        )  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(
            self.v_net(obs), -1
        )  # Critical to ensure v has right shape.
