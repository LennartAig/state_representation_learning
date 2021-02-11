import math

import torch
import torch.nn.functional as F


def infonce_loss(l, m):
    """Computes the noise contrastive estimation-based loss, a.k.a. infoNCE.
    Note that vectors should be sent as 1x1.
    Args:
        l: Local feature map.
        m: Multiple globals feature map.
    Returns:
        torch.Tensor: Loss.
    """
    N, units, n_locals = l.size()
    _, _, n_multis = m.size()

    # First we make the input tensors the right shape.
    l_p = l.permute(0, 2, 1)
    m_p = m.permute(0, 2, 1)

    l_n = l_p.reshape(-1, units)
    m_n = m_p.reshape(-1, units)

    # Inner product for positive samples. Outer product for negative. We need to do it this way
    # for the multiclass loss. For the outer product, we want a N x N x n_local x n_multi tensor.
    u_p = torch.matmul(l_p, m).unsqueeze(2)
    u_n = torch.mm(m_n, l_n.t())
    u_n = u_n.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    # We need to mask the diagonal part of the negative tensor.
    mask = torch.eye(N)[:, :, None, None].to(l.device)
    n_mask = 1 - mask

    # Masking is done by shifting the diagonal before exp.
    u_n = (n_mask * u_n) - (10.0 * (1 - n_mask))  # mask out "self" examples
    u_n = (
        u_n.reshape(N, N * n_locals, n_multis)
        .unsqueeze(dim=1)
        .expand(-1, n_locals, -1, -1)
    )

    # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
    pred_lgt = torch.cat([u_p, u_n], dim=2)
    pred_log = F.log_softmax(pred_lgt, dim=2)

    # The positive score is the first element of the log softmax.
    loss = -pred_log[:, :, 0].mean()

    return loss


def donsker_varadhan_loss(l, m):
    """
    Note that vectors should be sent as 1x1.
    Args:
        l: Local feature map.
        m: Multiple globals feature map.
    Returns:
        torch.Tensor: Loss.
    """
    N, units, n_locals = l.size()
    n_multis = m.size(2)

    # First we make the input tensors the right shape.
    l = l.view(N, units, n_locals)
    l = l.permute(0, 2, 1)
    l = l.reshape(-1, units)

    m = m.view(N, units, n_multis)
    m = m.permute(0, 2, 1)
    m = m.reshape(-1, units)

    # Outer product, we want a N x N x n_local x n_multi tensor.
    u = torch.mm(m, l.t())
    u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    # Since we have a big tensor with both positive and negative samples, we need to mask.
    mask = torch.eye(N).to(l.device)
    n_mask = 1 - mask

    # Positive term is just the average of the diagonal.
    E_pos = (u.mean(2) * mask).sum() / mask.sum()

    # Negative term is the log sum exp of the off-diagonal terms. Mask out the positive.
    u -= 10 * (1 - n_mask)
    u_max = torch.max(u)
    E_neg = (
        torch.log((n_mask * torch.exp(u - u_max)).sum() + 1e-6)
        + u_max
        - math.log(n_mask.sum())
    )
    loss = E_neg - E_pos

    return loss
