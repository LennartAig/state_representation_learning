import torch
import torch.nn as nn


class BaseModelSRL(nn.Module):
    """
    Base Class for a SRL network
    """

    def __init__(self):
        super(BaseModelSRL, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_state(self, obs, grad):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def optimize(self, loss):
        raise NotImplementedError
