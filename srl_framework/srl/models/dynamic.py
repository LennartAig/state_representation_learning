import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from srl_framework.utils.networks import make_mlp
from srl_framework.utils.distributions import Gaussian, Categorical


class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim, param, envtype, device=None):
        """
        Predict next state given current state and action
        
        Parameters:
        ------
            - state_dim (int):
            - action_dim (int):
        Return:
        ------
            - forward model (torch model): 
        """
        super(ForwardModel, self).__init__()
        self.deterministic = not (param.STOCHASTIC)
        self.envtype = envtype
        self.action_dim = action_dim

        self.mlp, dist_in_dim = make_mlp(
            input_dim=state_dim + action_dim,
            output_dim=0,
            architecture=[64, 64],
            activation="ReLU",
            output_layer=False,
            batchnorm=False,
            dropout=False,
            init="orthogonal",
        )
        self.dist = Gaussian(dist_in_dim, state_dim)

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=param.LEARNING_RATE)

    def forward(self, state, action, deterministic=True):
        """
        Predict next state given current state and action

        Inputs:
        ------
            - state (torch tensor)
            - action (torch tensor)
            - deterministic (bool):
        Return:
        ------
            - state_tp1_pred (torch tensor)
        """
        if self.envtype != "Box":
            action = F.one_hot(action, self.action_dim).float()
        if action.ndim < 2:
            action = action.unsqueeze(-1)
        concat = torch.cat((state, action), dim=1)
        pi = self.dist(self.mlp(concat))
        if deterministic:
            state_tp1_pred = pi.mode()
        else:
            state_tp1_pred = pi.rsample()
        return state_tp1_pred

    def predict_delta(self, state, action):
        """
        Predict the delta between the next state and current state
        
        Inputs:
        ------
            - state (torch tensor)
            - action (torch tensor)
        Return:
        ------
            - delta (torch tensor): diff between state and next state
        """
        return state - self(state, action)


class InverseModel(nn.Module):
    def __init__(self, state_dim, action_dim, envtype, param, device=None):
        """
        Predict action that transitions between given current state and
        consecutive state
        
        Parameters:
        ------
            - state_dim (int):
            - action_dim (int):
            - params
        Return:
        ------
            - inverse model (torch model): 
        """
        super(InverseModel, self).__init__()
        self.deterministic = not (param.STOCHASTIC)
        self.mlp, dist_in_dim = make_mlp(
            input_dim=state_dim + state_dim,
            output_dim=0,
            architecture=[64, 64],
            activation="ReLU",
            output_layer=False,
            batchnorm=False,
            dropout=False,
            init="orthogonal",
        )

        if envtype == "Discrete":
            self.dist = Categorical(dist_in_dim, action_dim)
        elif envtype == "Box":
            self.dist = Gaussian(dist_in_dim, action_dim)
        elif envtype == "MultiBinary":
            self.dist = Bernoulli(dist_in_dim, action_dim)
        else:
            raise NotImplementedError("Given env type is not known")

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=param.LEARNING_RATE)

    def forward(self, state, state_tp1, deterministic=True):
        """
        Predict action given current state and next state

        Inputs:
        ------
            - state (torch tensor)
            - state_tp1 (torchTensor)
        Return:
        ------
            - continuous: action
            - discrete: probability of each action (action logits)
        """
        concat = torch.cat((state, state_tp1), dim=1)
        pi = self.dist(self.mlp(concat))
        if deterministic:
            action_pred = pi.mode()
        else:
            action_pred = pi.rsample()
        return action_pred


class RewardModel(nn.Module):
    def __init__(self, state_dim, n_rewards, deterministic=False, lr=1e-3, device=None):
        """
        Predict reward given current state and next state
        Parameters:
        ------
            - input_dim (int):
            - n_rewards (int):
            - params
        Return:
        ------
            - reward model (torch model):
        """
        super(RewardModel, self).__init__()
        self.deterministic = deterministic
        self.mlp, dist_in_dim = make_mlp(
            input_dim=2 * state_dim,
            output_dim=0,
            architecture=[64, 64],
            activation="ReLU",
            output_layer=False,
            batchnorm=False,
            dropout=False,
            init="orthogonal",
        )
        self.dist = Gaussian(dist_in_dim, n_rewards)

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, state_tp1, deterministic=True):
        """
        Predict reward given input
        Parameters:
        ------
            - input (torch tensor): input to predict reward. If multiple (e.g. state and next state),
            concatinate them to form one tensor.
        Return:
        ------
            - predicted reward
        """
        concat = torch.cat((state, state_tp1), dim=-1)
        pi = self.dist(self.mlp(concat))
        if deterministic:
            rew_pred = pi.mode()
        else:
            rew_pred = pi.rsample()
        return rew_pred
