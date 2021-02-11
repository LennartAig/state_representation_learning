import torch
import torch.nn as nn
import math
import os 


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


# From https://github.com/pytorch/pytorch/pull/4411
def correlationMatrix(mat, eps=1e-8):
    """
    Returns Correlation matrix for mat. It is the equivalent of numpy np.corrcoef

    :param mat: (th.Tensor) Shape: (N, D)
    :param esp: (float) Small value to avoid division by zero.
    :return: (th.Tensor) The correlation matrix Shape: (N, N)
    """
    assert mat.dim() == 2, "Input must be a 2D matrix."
    mat_bar = mat - mat.mean(1).repeat(mat.size(1)).view(mat.size(1), -1).t()
    cov_matrix = mat_bar.mm(mat_bar.t()).div(mat_bar.size(1) - 1)
    inv_stddev = th.rsqrt(th.diag(cov_matrix) + eps)
    cor_matrix = cov_matrix.mul(inv_stddev.expand_as(cov_matrix))
    cor_matrix.mul_(inv_stddev.expand_as(cov_matrix).t())
    return cor_matrix.clamp(-1.0, 1.0)


def parameters_to_vector(parameters):
    r"""Convert parameters to one vector

    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)
        vec.append(param.reshape(-1))
        #vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_parameters(vec, parameters):
    r"""Convert one vector to the parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


def _check_param_device(param, old_param_device):
    r"""This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.

    Arguments:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device