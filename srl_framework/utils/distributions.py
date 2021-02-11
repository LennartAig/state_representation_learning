import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Modify standard PyTorch distributions so they are compatible with this code.
Taken from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/distributions.py
"""

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        log_prob = super().log_prob(actions.squeeze(-1))
        out = log_prob.view(actions.size(0), -1).sum(-1).unsqueeze(-1)
        return out

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(axis=-1, keepdim=True)  # keepdim ?!

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init_model(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        # TODO
        x = self.linear(x)
        return torch.distributions.Categorical(logits=x)


class Gaussian(nn.Module):
    def __init__(
        self, num_inputs, num_outputs, fixed_std=True, log_std_min=-20, log_std_max=2
    ):
        super(Gaussian, self).__init__()

        self.fixed_std = fixed_std

        init_ = lambda m: init_model(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.mu_layer = init_(nn.Linear(num_inputs, num_outputs))
        if self.fixed_std:
            log_std = -0.5 * np.ones(num_outputs, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        else:
            self.log_std_layer = init_(nn.Linear(num_inputs, num_outputs))
            self.log_std_min = log_std_min
            self.log_std_max = log_std_max

    def forward(self, x):
        mu = self.mu_layer(x)
        if self.fixed_std:
            log_std = self.log_std
        else:
            log_std = self.log_std_layer(x)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        std = torch.exp(log_std)
        return FixedNormal(mu, std)


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init_model(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


def init_model(module, weight_init, bias_init, gain=1):
    """
    Helper function to specifically initialize models.
    Orginal here: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/utils.py
    Pytorch init functions that can be used: https://pytorch.org/docs/stable/nn.init.html
    
    Parameters:
    -----------
        - module: pytorch nn.module 
        - weight_init: pytorch nn.init function
        - bias_init: pytorch nn.init function
        - gain: float
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class AddBias(nn.Module):
    """
    Helper class to do KFAC
    Orginal here: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/utils.py
    
    Parameters:
    -----------
        - bias: pytorch nn.init function
    """

    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
