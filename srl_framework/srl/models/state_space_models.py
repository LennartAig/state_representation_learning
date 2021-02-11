import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class BaseRecurrentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_dim, seq_len, device):
        """
        Predict next state given current state and action
        
        Parameters:
        ------
            - input_dim (int):
            - hidden_dim (int):
        Return:
        ------
            - recurrent model (torch model): 
        """
        super(BaseRecurrentModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.device = device

    def forward(self, state, action):
        """
        Predict next state given current state and action

        Inputs:
        ------
            - state (torch tensor)
            - action (torch tensor)
        Return:
        ------
            - state_tp1 (torch tensor)
        """

        return self.forward_net(concat)


class WorldModel(BaseRecurrentModel):
    def __init__(
        self, state_dim, action_dim, hidden_dim, num_layers_rnn, device, seq_len=2
    ):
        """        
        Parameters:
        ------
            - state_dim (int): State dim could be action+state_dim
            - action_dim (int)
            - params
        """
        super(WorldModel, self).__init__(
            state_dim + action_dim, hidden_dim, state_dim, seq_len, device
        )
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.recurrent_net = nn.LSTM(
            hidden_dim, hidden_dim, num_layers_rnn, batch_first=True
        )
        self.fc2 = nn.Linear(hidden_dim, state_dim)
        self.rnn_hidden_dim = hidden_dim
        self.num_layers = num_layers_rnn
        self.hidden = (
            torch.randn(self.num_layers, 1, self.hidden_dim, device=self.device),
            torch.randn(self.num_layers, 1, self.hidden_dim, device=self.device),
        )

        self.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, state, action):
        """
        Predict action given current state

        Inputs:
        ------
            - state (list of torch tensors)
            - action (list of torch tensors)
        Return:
        ------
            - 
        """
        state = torch.cat([state, action], dim=-1)  # .unsqueeze(0)
        if state.ndim < 3:
            state = state.unsqueeze(0)
        state = self.fc1(state)
        h, _ = self.recurrent_net(
            state
        )  # if rnn_hidden == None else self.recurrent_net(state)
        state_tp1_pred = self.fc2(h)
        return state_tp1_pred, 0

    def infer(self, state, action, rnn_hidden=None):
        """
        Return hidden state 

        Inputs:
        ------
            - state (deque)
            - action (deque)
            - rnn_hidden
        Return:
        ------
            - 
        """
        if rnn_hidden:
            rnn_hidden = self.hidden
        if action.ndim < 2:
            action = action.unsqueeze(0)
        state = torch.cat([state, action], dim=-1)
        if state.ndim < 3:
            state = state.unsqueeze(0)
        state = self.fc1(state)
        h, hidden = self.recurrent_net(state)
        return hidden

    def init_hidden(self):
        """
        """
        self.hidden = (
            torch.randn(self.num_layers, 1, self.hidden_dim, device=self.device),
            torch.randn(self.num_layers, 1, self.hidden_dim, device=self.device),
        )


class RSSM(nn.Module):
    """
    This class includes multiple components
    Deterministic state model: h_t+1 = f(h_t, s_t, a_t)
    Stochastic state model (prior): p(s_t+1 | h_t+1)
    State posterior: q(s_t | h_t, o_t)
    https://github.com/cross32768/PlaNet_PyTorch/blob/master/model.py
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        rnn_hidden_dim,
        device=None,
        hidden_dim=200,
        min_stddev=0.1,
        act=F.relu,
    ):
        super(RSSM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc_state_action = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc_rnn_hidden = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc_state_mean_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_prior = nn.Linear(hidden_dim, state_dim)
        # Size of embedded image can be bigger! TODO
        self.fc_rnn_hidden_embedded_obs = nn.Linear(
            rnn_hidden_dim + state_dim, hidden_dim
        )
        self.fc_state_mean_posterior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_posterior = nn.Linear(hidden_dim, state_dim)
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self._min_stddev = min_stddev
        self.act = act
        # feat(t) = x(t) : This encoding is performed deterministically.
        self.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state, action, rnn_hidden, embedded_next_obs):
        """
        h_t+1 = f(h_t, s_t, a_t)
        Return prior p(s_t+1 | h_t+1) and posterior p(s_t+1 | h_t+1, o_t+1)
        for model training
        """
        next_state_prior, rnn_hidden = self.prior(state, action, rnn_hidden)
        next_state_posterior = self.posterior(rnn_hidden, embedded_next_obs)
        return next_state_prior, next_state_posterior, rnn_hidden

    def prior(self, state, action, rnn_hidden):
        """
        h_t+1 = f(h_t, s_t, a_t)
        Compute prior p(s_t+1 | h_t+1)
        """
        hidden = self.act(self.fc_state_action(torch.cat([state, action], dim=1)))
        rnn_hidden = self.rnn(hidden, rnn_hidden)
        hidden = self.act(self.fc_rnn_hidden(rnn_hidden))

        mean = self.fc_state_mean_prior(hidden)
        stddev = F.softplus(self.fc_state_stddev_prior(hidden)) + self._min_stddev
        return Normal(mean, stddev), rnn_hidden

    def posterior(self, rnn_hidden, state):
        """
        Compute posterior q(s_t | h_t, o_t)
        """
        hidden = self.act(
            self.fc_rnn_hidden_embedded_obs(torch.cat([rnn_hidden, state], dim=1))
        )
        mean = self.fc_state_mean_posterior(hidden)
        stddev = F.softplus(self.fc_state_stddev_posterior(hidden)) + self._min_stddev
        return Normal(mean, stddev)

    def infer(self, state, action, rnn_hidden):
        """
        Return hidden state

        Inputs:
        ------
            - state
            - action
        Return:
        ------
            - hidden
        """
        _, rnn_hidden = self.prior(state, action, rnn_hidden)
        return rnn_hidden

    def init_hidden(self):
        """
        """
        return torch.zeros(1, self.rnn_hidden_dim, device=self.device)
