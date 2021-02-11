import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from srl_framework.utils.encoder import CnnEncoder, ResNetEncoder, PixelEncoder
from srl_framework.utils.networks import make_mlp
from srl_framework.utils.distributions import Gaussian

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
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class SquashedGaussianPolicy(nn.Module):
    def __init__(self, encoder, log_std_min=-10, log_std_max=2,**kwargs):
        super(SquashedGaussianPolicy, self).__init__()
        self.act_limit = kwargs['action_limit']
        self.use_encoder = kwargs['use_cnn']
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        if kwargs['use_cnn']:
            if encoder:
                self.encoder = encoder
            else:
                encoder_args = kwargs['encoder_args']
                if encoder_args['architecture'] == 'impala':
                    self.encoder = ResNetEncoder(
                        img_channels = kwargs['img_channels'], feature_dim=kwargs['state_dim'],
                        img_size = kwargs['img_size'], normalized_obs=kwargs['normalized_obs'],
                        squash_latent=encoder_args['squash_latent'], normalize=encoder_args['normalize'],
                        conv_init=encoder_args['conv_init'], linear_init=encoder_args['linear_init'])
                else:
                    self.encoder = CnnEncoder(
                        img_channels = kwargs['img_channels'], feature_dim=kwargs['state_dim'],
                        img_size = kwargs['img_size'], normalized_obs=kwargs['normalized_obs'],
                        architecture=encoder_args['architecture'],
                        squash_latent=encoder_args['squash_latent'], normalize=encoder_args['normalize'],
                        conv_init=encoder_args['conv_init'], linear_init=encoder_args['linear_init'],
                        activation=encoder_args['activation'], batchnorm=encoder_args['batchnorm'], 
                        pool=encoder_args['pool'], dropout=encoder_args['dropout'])
            
        self.vae = True if kwargs['latent_type'] == 'vae' else False

        self.mlp, dist_in_dim = make_mlp(input_dim=kwargs['input_dim_actor'],
            output_dim=0, architecture= [64,64], activation = "ReLU", output_layer=False,
            batchnorm=False, dropout=False, init = "orthogonal")
        self.dist = Gaussian(dist_in_dim, kwargs['action_dim'], fixed_std=False)
    
    def forward(self, state, deterministic=False, log_prob = True, detach_encoder = False):
        features = self.encoder(state, detach = detach_encoder) if self.use_encoder else state
        #if self.use_encoder and self.vae:
        #    features,_ = features.chunk(2, dim=-1) # take mean as input
        features = self.mlp(features) 
        pi = self.dist(features)
        action = pi.mode() if deterministic else pi.rsample()
        if log_prob:
            log_prob = pi.log_probs(action)
            log_prob -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1, keepdim=True)
        else:
            log_prob = None
        action = torch.tanh(action) * self.act_limit
        return action, log_prob


class DoubleActionValueFunction(nn.Module):
    """
    Double State Action Value Function:
    ----------------------
    State Action Value Function takes state as input and returns two
    q values: Q_1(s,a), Q_2(s,a)

    Use in continous action spaces and algorithms such as SAC.
    """
    def __init__(self,encoder=None,**kwargs):
        super(DoubleActionValueFunction, self).__init__()
        self.use_encoder = kwargs['use_cnn']
        if kwargs['use_cnn']:
            if encoder:
                self.encoder = encoder
            else:
                encoder_args = kwargs['encoder_args']
                if encoder_args['architecture'] == 'impala':
                    self.encoder = ResNetEncoder(
                        img_channels = kwargs['img_channels'], feature_dim=kwargs['feature_dim'],
                        img_size = kwargs['img_size'], normalized_obs=kwargs['normalized_obs'],
                        squash_latent=encoder_args['squash_latent'], normalize=encoder_args['normalize'],
                        conv_init=encoder_args['conv_init'], linear_init=encoder_args['linear_init'])
                elif encoder_args['architecture'] == 'standard':
                    self.encoder = PixelEncoder((kwargs['img_channels'],kwargs['img_size'],kwargs['img_size']),kwargs['feature_dim'])
                else:
                    self.encoder = CnnEncoder(
                        img_channels = kwargs['img_channels'], feature_dim=kwargs['feature_dim'],
                        img_size = kwargs['img_size'], normalized_obs=kwargs['normalized_obs'],
                        architecture=encoder_args['architecture'],
                        squash_latent=encoder_args['squash_latent'], normalize=encoder_args['normalize'],
                        conv_init=encoder_args['conv_init'], linear_init=encoder_args['linear_init'],
                        activation=encoder_args['activation'], batchnorm=encoder_args['batchnorm'], 
                        pool=encoder_args['pool'], dropout=encoder_args['dropout'])
                kwargs['input_dim_critic'] = kwargs['feature_dim']
                kwargs['input_dim_actor'] = kwargs['feature_dim']
        self.vae = True if kwargs['latent_type'] == 'vae' else False

        self.q1 = StateActionValueFunction(**kwargs)
        self.q2 = StateActionValueFunction(**kwargs)
        self.apply(weight_init)

    def forward(self, state, action,detach_encoder = False):
        features = self.encoder(state, detach = detach_encoder) if self.use_encoder else state
        #if self.use_encoder and self.vae:
        #    features,_ = features.chunk(2, dim=-1) # take mean as input
        q1 = self.q1(features, action)
        q2 = self.q2(features, action)
        return q1, q2

class StateActionValueFunction(nn.Module):
    def __init__(self, **kwargs):
        super(StateActionValueFunction, self).__init__()
        self.mlp, _ = make_mlp(input_dim=kwargs['input_dim_critic']+kwargs['action_dim'],
            output_dim=1, architecture= [64,64], activation = "ReLU", output_layer=True,
            batchnorm=False, dropout=False, init = "orthogonal")

    def forward(self, state, action):
        q_value = self.mlp(torch.cat([state, action], dim=1))
        return q_value
        #return torch.squeeze(q_value, -1) 