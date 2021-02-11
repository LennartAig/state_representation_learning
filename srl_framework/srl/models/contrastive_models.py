import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from copy import deepcopy
import utils

from srl_framework.utils.encoder import CnnEncoder, ResNetEncoder, PixelEncoder



class CURL(nn.Module):
    """
    CURL
    """

    def __init__(self, state_dim, output_type="continuous", polyak = 0.99, img_channels=3,
                encoder_args=None, lr=0.001, device=None, img_size=84, normalized_obs=True):
        super(CURL, self).__init__()

        self.polyak = polyak

        if encoder_args['architecture'] == 'impala':
            self.encoder = ResNetEncoder(
                img_channels = img_channels, feature_dim=state_dim, img_size = img_size, normalized_obs=normalized_obs,
                squash_latent=encoder_args['squash_latent'], normalize=encoder_args['normalize'],
                conv_init=encoder_args['conv_init'], linear_init=encoder_args['linear_init'])
        elif encoder_args['architecture'] == 'standard':
            self.encoder = PixelEncoder((img_channels,img_size,img_size),state_dim)
        else:
            self.encoder = CnnEncoder(
                img_channels = img_channels, feature_dim=state_dim, img_size = img_size,
                architecture=encoder_args['architecture'], normalized_obs=normalized_obs,
                squash_latent=encoder_args['squash_latent'], normalize=encoder_args['normalize'],
                conv_init=encoder_args['conv_init'], linear_init=encoder_args['linear_init'],
                activation=encoder_args['activation'], batchnorm=encoder_args['batchnorm'], 
                pool=encoder_args['pool'], dropout=encoder_args['dropout'])

        self.encoder_targ = deepcopy(self.encoder)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        #for p in self.encoder_targ.parameters():
        #    p.requires_grad = False

        self.W = nn.Parameter(torch.rand(state_dim, state_dim))
        self.output_type = output_type
        self.device = device
        self.to(self.device)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.003)
        self.train()
        self.encoder_targ.train()
    
    def get_state(self, x, detach=False):
        return self.encode(x,detach=detach)

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        Inputs:
        ------
            - :param x: x_t, x y coordinates
        Returns:
        ------
            - :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_targ(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits
    
    def get_loss(self, obs_anchor, obs_pos):
        
        z_a = self.encode(obs_anchor)
        z_pos = self.encode(obs_pos, ema=True)
        
        logits = self.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        cross_entropy_loss = F.cross_entropy(logits, labels)

        return cross_entropy_loss
    
    def target_update(self):
        """
        Update of target networks by polyak averaging
        """
        with torch.no_grad():
            for p, p_targ in zip(self.encoder.parameters(), self.encoder_targ .parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)