from __future__ import print_function, division, absolute_import

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl as kl


def l1_loss(prediction, goal):
    lossfct = nn.L1Loss()
    return lossfct(prediction, goal)


def robotic_prior_loss(states, next_states, actions, rewards):
    """
    TODO: Disentangle robotic priors to better guide loss.
    Computing the 4 Robotic priors: Temporal coherence, Causality, Proportionality, Repeatability
    Input:
    ------
        - states (torch tensor)
        - next_states (torch tensor)
        - actions (torch tensor): 
        - rewards (torch tensor): 
    Return: 
        - loss (torch tensor)
    """
    dissimilar_pairs = find_dissimilar(actions, rewards, action_limits=None)
    same_actions_pairs = find_same_actions(actions)
    # Temporal coherence
    state_diff = next_states - states
    state_diff_norm = state_diff.norm(2, dim=1)
    temp_coherence_loss = (state_diff_norm ** 2).mean()

    # Causality loss
    similarity = lambda x, y: torch.exp(-(x - y).norm(2, dim=1) ** 2)
    causality_loss = similarity(
        states[dissimilar_pairs[:, 0]], states[dissimilar_pairs[:, 1]]
    ).mean()
    proportionality_loss = (
        (
            state_diff_norm[same_actions_pairs[:, 0]]
            - state_diff_norm[same_actions_pairs[:, 1]]
        )
        ** 2
    ).mean()

    repeatability_loss = (
        similarity(states[same_actions_pairs[:, 0]], states[same_actions_pairs[:, 1]])
        * (
            state_diff[same_actions_pairs[:, 0]] - state_diff[same_actions_pairs[:, 1]]
        ).norm(2, dim=1)
        ** 2
    ).mean()
    losses = {
        "temp_coherence_loss": temp_coherence_loss,
        "causality_loss": causality_loss,
        "proportionality_loss": proportionality_loss,
        "repeatability_loss": repeatability_loss,
    }
    return losses


def position_velocity_loss(states, next_states, actions, rewards):
    """
    Computing the 4 Robotic priors: Temporal coherence, Causality, Proportionality, Repeatability
    Input:
    ------
        - states (torch tensor)
        - next_states (torch tensor)
        - actions (torch tensor): 
        - rewards (torch tensor): 
    Return: 
        - loss (torch tensor)
    """
    velocity = alpha * (next_states * states)
    dissimilar_pairs = find_dissimilar(actions, rewards, action_limits=None)
    same_actions_pairs = find_same_actions(actions)
    # Temporal coherencedef calc_latent_loss(self, images_seq, actions_seq, rewards_seq,dones_seq):
    proportionality_loss = (
        (
            state_diff_norm[same_actions_pairs[:, 0]]
            - state_diff_norm[same_actions_pairs[:, 1]]
        )
        ** 2
    ).mean()

    repeatability_loss = (
        similarity(states[same_actions_pairs[:, 0]], states[same_actions_pairs[:, 1]])
        * (
            state_diff[same_actions_pairs[:, 0]] - state_diff[same_actions_pairs[:, 1]]
        ).norm(2, dim=1)
        ** 2
    ).mean()
    weights = [1, 1, 1, 1]
    names = [
        "temp_coherence_loss",
        "causality_loss",
        "proportionality_loss",
        "repeatability_loss",
    ]
    losses = [
        temp_coherence_loss,
        causality_loss,
        proportionality_loss,
        repeatability_loss,
    ]

    total_loss = 0
    for idx in range(len(weights)):
        total_loss += losses[idx] * weights[idx]
    return causality_loss


def forward_loss(state_tp1_pred, state_tp1):
    """
    Input:
    ------
        - state_tp1_pred (torch tensor):
        - state_tp1 (torch tensor):
    Return:
    ------
    """
    return reconstruction_loss(next_states_pred, next_states)


def invers_loss(actions_pred, actions, envtype):
    """
    Inverse model's loss: 
    Cross-entropy between predicted categoriacal actions and true actions
    Input:
    -----
        - actions_pred: torch tensor
        - actions_st: torch tensor
        - envtype (string): 'discrete' -> cross-entropy-loss
                            'continuous' -> mse-loss 
    Return:
    ------
    """
    if envtype == "discrete":
        inverse_loss = F.cross_entropy(actions_pred, actions)
    elif envtype == "continuous":
        inverse_loss = F.mse_loss(actions_pred, actions)
    else:
        raise "Inverse loss cannot be computed. Unknown action type"

    return inverse_loss


def l1_regularization(network_params):
    """
    L1 regularization
    Input:
    ------
        - network_params (torch tensor): NN's weights to regularize
    Return:
    ------
        - loss (torch tensor)
    """
    l1_loss = sum([param.norm(1) for param in network_params]) / len(network_params)
    return l1_loss


def l2_regularization(network_params):
    """
    L2 regularization
    Input:
    ------
        - network_params (torch tensor): NN's weights to regularize
    Return:
    ------
        - loss (torch tensor)
    """
    l2_loss = sum([param.norm(2) for param in network_params]) / len(network_params)
    return l2_loss


def latent_regularization_loss(state):
    """
    L2 Regularization of latent space (here state space) of Autoencoder
    (see https://arxiv.org/pdf/1903.12436.pdf)
    Input:
    ------
        - state (torch tensor): 
    """
    return (0.5 * state.pow(2).sum(1)).mean()


def reward_loss(rewards_pred, rewards_st):
    """
    Categorical Reward prediction Loss (Cross-entropy)
    Input:
    ------
        - rewards_pred (torch tensor): predicted reward - categorical 
        - rewards_st (torch tensor)
    Return:
    ------
        - 
    """
    return nn.CrossEntropyLoss(rewards_pred, target=rewards_st)


def log_liklihood_from_dist(dist, label):
    """
    Input:
    ------
        - dist (torch distribution): distribtion
        - label (torch tensor): observations as labels for log liklihood loss
    """
    log_likelihood_loss = dist.log_prob(label).mean(dim=0)
    return log_likelihood_loss


def log_liklihood(decoded_obs_t, obs_t, std):
    """
    Input:
    ------
        - decoded_obs_t (torch tensor): decoded observations as mean of normal distribtion
        - std (torch tensor): standard deviation of normal distribtion
        - obs_t (torch tensor): observations as labels for log liklihood loss
    """
    if decoded_obs_t.ndim > 4:
        flag = True
        num_batches, num_sequences, C, H, W = decoded_obs_t.size()
        decoded_obs_t = decoded_obs_t.view(num_batches * num_sequences, C, H, W)
        obs_t = obs_t.view(num_batches * num_sequences, C, H, W)
    decoded_obs_dist = Normal(decoded_obs_t, torch.ones_like(decoded_obs_t) * std)
    return log_liklihood_from_dist(decoded_obs_dist, obs_t)


def reconstruction_loss(input_image, target_image):
    """
    Reconstruction Loss (Mean Squared Error) for Autoencoders
    Input:
    ------
        - input_image (torch tensor): Observation 
        - target_image (torch tensor): Reconstructed observation 
    Return:
    ------
        - mean squared error loss
    """
    return F.mse_loss(input_image, target_image)


def calculate_kl_divergence(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow_(2)
    t1 = ((p_mean - q_mean) / q_std).pow_(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


def rae_loss(obs_t, decoded_obs_t, state_t, latent_lambda):
    """
    Regul
    see https://arxiv.org/pdf/1903.12436.pdf
    Input
    ------
        - obs_t (torch tensor): Observation 
        - decoded_obs_t (torch tensor): reconstructed Observation
        - state_t (torch tensor): state representation of obs_t in latent space
        - latent_lambda (float): weighting factor
    Return:
    ------
        - loss (torch tensor)
    """
    rec_loss = F.mse_loss(decoded_obs_t, obs_t)

    # Add L2 penalty on latent representation
    latent_loss = (0.5 * state_t.pow(2).sum(1)).mean()

    loss = rec_loss + latent_lambda * latent_loss
    return loss


def generationLoss(obs_t, decoded_obs_t):
    """
    Pixel-wise generation Loss
    Input
    ------
        - obs_t (torch tensor): Observation 
        - decoded_obs_t (torch tensor): reconstructed Observation
    Return:
    ------
        - loss (torch tensor)
    """
    generation_loss = F.mse_loss(decoded_obs_t, obs_t, reduction="sum")
    return generation_loss


def perceptual_similarity_loss(encoded_real, encoded_prediction):
    """
    Perceptual similarity Loss for VAE as in
    "DARLA: Improving Zero-Shot Transfer in Reinforcement Learning", Higgins et al.
    see https://arxiv.org/pdf/1707.08475.pdf
    Input:
    ------
        - encoded_real: states encoding the real observation by the DAE (torch tensor)
        - encoded_prediction: states encoding the vae's predicted observation by the DAE  (torch tensor)

    Return:
    ------ 
        - (torch tensor)
    """

    pretrained_dae_encoding_loss = F.mse_loss(
        encoded_real, encoded_prediction, reduction="sum"
    )
    return pretrained_dae_encoding_loss


def kl_loss(mu, logvar):
    """
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 (https://arxiv.org/abs/1312.6114)

    KL divergence losses summed over all elements and batch
    Input:
    ------
        - mu (torch tensor): mean of the distribution of samples 
        - logvar (torch tensor): Logarithm of variance of the distribution of samples 
    Return:
    ------
        - KL Divergence  
    """
    mu_sq = mu * mu
    sigma = torch.exp(logvar)
    sigma_sq = sigma * sigma
    kl_divergence = 0.5 * torch.mean(mu_sq + sigma_sq - torch.log(sigma_sq) - 1)
    return kl_divergence


def kl_loss(mu, logvar, beta):
    """
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014 (https://arxiv.org/abs/1312.6114)

    KL divergence losses summed over all elements and batch
    Input:
    ------
        - mu (torch tensor): mean of the distribution of samples 
        - logvar (torch tensor): Logarithm of variance of the distribution of samples 
        - beta (float):  used to weight the KL divergence for disentangling
    Return:
    ------
        - Negative KL Divergence  
    """
    return torch.mean(
        -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0
    )


def kl_divergence_seq(p, q):
    """
    KL divergence losses summed over batches of sequetial data
    Input:
    ------
        - p (list [len_seq] of torch.dist.Normal[batch, statedim])
        - q (list [len_seq] of torch.dist.Normal[batch, statedim])
    Return:
    ------
        - Negative KL Divergence  
    """
    assert len(p) == len(q)

    kld = 0.0
    for i in range(len(p)):
        # (N, L) shaped array of kl divergences.
        _kld = kl.kl_divergence(p[i], q[i])
        # Average along batches, sum along sequences and elements.
        kld += _kld.mean(dim=0).sum()
    return kld


def cross_entropy(logits, labels):
    cross_entropy_loss = F.cross_entropy(logits, labels)
    return cross_entropy_loss


def mutual_unformation_loss(state_t, rewards):
    """
    Loss criterion to assess mutual information between predicted states and rewards
    see: https://en.wikipedia.org/wiki/Mutual_information
    Inputs:
    ------
        - states (torch tensor):
        - rewards (torch tensor):
    Return:
    ------
        - loss (torch tensor)
    """
    X = state_t
    Y = rewards
    I = 0
    eps = 1e-10
    p_x = (
        float(1 / np.sqrt(2 * np.pi))
        * th.exp(
            -th.pow(
                th.norm((X - th.mean(X, dim=0)) / (th.std(X, dim=0) + eps), 2, dim=1), 2
            )
            / 2
        )
        + eps
    )
    p_y = (
        float(1 / np.sqrt(2 * np.pi))
        * th.exp(
            -th.pow(
                th.norm((Y - th.mean(Y, dim=0)) / (th.std(Y, dim=0) + eps), 2, dim=1), 2
            )
            / 2
        )
        + eps
    )
    for x in range(X.shape[0]):
        for y in range(Y.shape[0]):
            p_xy = (
                float(1 / np.sqrt(2 * np.pi))
                * th.exp(
                    -th.pow(
                        th.norm(
                            (
                                th.cat([X[x], Y[y]])
                                - th.mean(th.cat([X, Y], dim=1), dim=0)
                            )
                            / (th.std(th.cat([X, Y], dim=1), dim=0) + eps),
                            2,
                        ),
                        2,
                    )
                    / 2
                )
                + eps
            )
            I += p_xy * th.log(p_xy / (p_x[x] * p_y[y]))

    mutual_info_loss = th.exp(-I)
    return mutual_info_loss


def calc_latent_loss(self, images_seq, actions_seq, rewards_seq, dones_seq):
    # TODO
    features_seq = self.latent.encoder(images_seq)

    # Sample from posterior dynamics.
    (
        (latent1_post_samples, latent2_post_samples),
        (latent1_post_dists, latent2_post_dists),
    ) = self.latent.sample_posterior(features_seq, actions_seq)
    # Sample from prior dynamics.
    (
        (latent1_pri_samples, latent2_pri_samples),
        (latent1_pri_dists, latent2_pri_dists),
    ) = self.latent.sample_prior(actions_seq)

    # KL divergence loss.
    kld_loss = calc_kl_divergence(latent1_post_dists, latent1_pri_dists)

    # Log likelihood loss of generated observations.
    images_seq_dists = self.latent.decoder([latent1_post_samples, latent2_post_samples])
    log_likelihood_loss = images_seq_dists.log_prob(images_seq).mean(dim=0).sum()

    # Log likelihood loss of genarated rewards.
    rewards_seq_dists = self.latent.reward_predictor(
        [
            latent1_post_samples[:, :-1],
            latent2_post_samples[:, :-1],
            actions_seq,
            latent1_post_samples[:, 1:],
            latent2_post_samples[:, 1:],
        ]
    )
    reward_log_likelihoods = rewards_seq_dists.log_prob(rewards_seq) * (1.0 - dones_seq)
    reward_log_likelihood_loss = reward_log_likelihoods.mean(dim=0).sum()

    latent_loss = kld_loss - log_likelihood_loss - reward_log_likelihood_loss

    return latent_loss


# ---------------------------------------------------------------- #
#                               UTILS                              #
# ---------------------------------------------------------------- #


def find_dissimilar(actions, rewards, action_limits):
    """
    Finds indicies of states which should be dissimilar because the same action led to 
    different reward.
    Input:
    ------
        - actions (torch tensor [minibatch, action_size]): action minibatch
        - rewards (torch tensor [minibatch]): reward minibatch
        - action limits (np array [action_size]): action limit to set a range of actions
                                                which are close enough to 
    Return:
    ------
        - indices
    """
    empty = True
    rewards_np = rewards.cpu().numpy()
    actions_np = actions.cpu().numpy()
    # Normalize rewards
    rewards_min, rewards_max = np.min(rewards_np), np.max(rewards_np)
    if np.min(rewards_np) <= 0:
        rewards_normalized = (rewards_np + rewards_min) / (rewards_min + rewards_max)
    else:
        rewards_normalized = (rewards_np - rewards_min) / (-rewards_min + rewards_max)

    for i, action in enumerate(actions_np):
        positions = np.where(
            (actions_np > (action - 0.02))
            * (actions_np < (action + 0.02))
            * (rewards_normalized < rewards_normalized[i] - 0.1)
            + (rewards_normalized > rewards_normalized[i] + 0.1)
        )[0]
        positions = positions[np.where(positions > i)]
        if positions.size > 0:
            pairs = np.empty((positions.size, 2), dtype=int)
            pairs[:, 0] = positions
            pairs[:, 1] = np.ones_like(positions, dtype=int) * np.array(i, dtype=int)
            if empty:
                dissimilar_pairs = pairs
                empty = False
            else:
                dissimilar_pairs = np.append(dissimilar_pairs, pairs, axis=0)
    return dissimilar_pairs


def find_same_actions(actions):
    """
    Get observations indices where the same action was performed as in a reference observation
    Input:
    ------
        - actions (torch tensor [minibatch, action_size]): action minibatch
        - action limits (np array [action_size]): action limit to set a range of actions
                                                which are close enough to 
    Return:
    ------
        - indices
    """
    empty = True
    similar_pairs = np.array([])
    actions_np = actions.cpu().numpy()
    for i, action in enumerate(actions_np):
        positions = np.where(
            (actions_np > (action - 0.02)) * (actions_np < (action + 0.02))
        )[0]
        positions = positions[np.where(positions > i)].astype(int)
        if positions.size > 0:
            pairs = np.empty((positions.size, 2), dtype=int)
            pairs[:, 0] = positions
            pairs[:, 1] = np.ones_like(positions, dtype=int) * np.array(i, dtype=int)
            if empty:
                similar_pairs = pairs
                empty = False
            else:
                similar_pairs = np.append(similar_pairs, pairs, axis=0)
    return similar_pairs
