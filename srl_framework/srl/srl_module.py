import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.distributions.kl import kl_divergence
import math

from srl_framework.srl.models.information_maximization import (
    ST_DIM_Loss,
    DIM_Loss,
    JSD_ST_DIM_Loss,
)
from srl_framework.srl.models import (
    VAE,
    AE,
    CURL,
    LatentNetwork,
    ForwardModel,
    InverseModel,
    RewardModel,
    WorldModel,
    RSSM,
    DeepInfoMax,
    DeepInfoMaxLoss,
    Encoder,
)
from srl_framework.srl.losses.losses import (
    kl_loss,
    reconstruction_loss,
    rae_loss,
    robotic_prior_loss,
    l1_loss,
    calculate_kl_divergence,
)
from srl_framework.srl.utils import preprocess_obs
from copy import deepcopy


def isNaN(num):
    return num != num


class SRLModule(nn.Module):
    """
    """

    def __init__(self, param, **kwargs):
        super(SRLModule, self).__init__()

        self.info = {
            "use_srl": False,
            "state_type": "observation",
            "sequential": False,
            "seq_len": 0,
            "latent": False,
            "stochastic_model": False,
            "contrastive": False,
            "rl_loss": False,
            "state_dim": 0,
            "input_dim_critic": 0,
            "input_dim_actor": 0,
            "safe_hidden": False,
            "use_cnn": True,
            "method": "none",
        }
        self.param = param
        self.obs_shape = kwargs["obs_shape"]
        self.info["state_dim"] = (
            self.obs_shape[0] if len(self.obs_shape) == 1 else self.obs_shape
        )
        self.info["use_cnn"] = False if len(self.obs_shape) == 1 else True
        self.act_dim = kwargs["act_dim"]
        self.envtype = kwargs["envtype"]
        self.device = kwargs["device"]
        self.normalized_obs = kwargs["normalized_obs"]
        self.info["data_augmentation"] = False
        self.encoder_args = kwargs["encoder_args"]
        self.info["latent_type"] = None
        self.grad_clip = False
        self.do_target_update = False

        self.loss_weights = {}
        self.losses = {}

        if self.param.USE:
            self.info["use_srl"] = True
            self.info["data_augmentation"] = (
                True if "DRQ" in self.param.MODELS else False
            )
            if "LATENT" in self.param.MODELS:
                self.train_latent = True

                if len(self.obs_shape) == 1:
                    raise Exception(
                        "Not possible to use laten model with non pixel state space"
                    )

                print("Latent Model: {}".format(self.param.LATENT.TYPE))

                self.info["latent"] = True

                self.info["use_cnn"] = True
                self.info["state_dim"] = self.param.LATENT.STATESIZE
                self.info["input_dim_actor"] = self.info["state_dim"]
                self.info["input_dim_critic"] = self.info["state_dim"]

                if "SSM" in self.param.MODELS:
                    self.decoder_dim = (
                        self.param.SSM.LATENT_2_DIM + self.param.SSM.LATENT_1_DIM
                    )
                else:
                    self.decoder_dim = self.info["state_dim"]

                # Type of autoencoder/latent model
                if self.param.LATENT.TYPE == "AE" and self.train_latent:
                    self.latent_model = AE(
                        img_channels=self.obs_shape[0],
                        state_dim=self.info["state_dim"],
                        encoder_args=kwargs["encoder_args"],
                        lr=0.001,
                        decoder_dim=self.decoder_dim,
                        img_size=self.obs_shape[-1],
                        normalized_obs=kwargs["normalized_obs"],
                    )
                    self.loss_weights[
                        "reconstruction_loss"
                    ] = self.param.LATENT.LOSS_WEIGHT
                    self.losses["reconstruction_loss"] = 0
                elif self.param.LATENT.TYPE == "RAE" and self.train_latent:
                    self.latent_model = AE(
                        img_channels=self.obs_shape[0],
                        state_dim=self.info["state_dim"],
                        encoder_args=kwargs["encoder_args"],
                        lr=0.001,
                        decoder_dim=self.decoder_dim,
                        img_size=self.obs_shape[-1],
                        normalized_obs=kwargs["normalized_obs"],
                    )
                    self.beta = 1e-6
                    self.loss_weights["rae_loss"] = self.param.LATENT.LOSS_WEIGHT
                    self.losses["rae_loss"] = 0
                elif self.param.LATENT.TYPE == "VAE":
                    self.latent_model = VAE(
                        img_channels=self.obs_shape[0],
                        state_dim=self.info["state_dim"],
                        encoder_args=kwargs["encoder_args"],
                        lr=0.001,
                        decoder_dim=self.decoder_dim,
                        img_size=self.obs_shape[-1],
                        normalized_obs=kwargs["normalized_obs"],
                    )
                    self.beta = 1.0
                    self.loss_weights["reconstruction_loss"] = 1
                    self.losses["reconstruction_loss"] = 0
                    self.loss_weights["kl_divergence"] = 1e-6
                    self.losses["kl_divergence"] = 0
                    self.info["latent_type"] = "vae"
                elif self.param.LATENT.TYPE == "ENCODER":
                    # No use of decoder later on
                    self.latent_model = Encoder(
                        img_channels=self.obs_shape[0],
                        state_dim=self.info["state_dim"],
                        encoder_args=kwargs["encoder_args"],
                        lr=0.001,
                        img_size=self.obs_shape[-1],
                        normalized_obs=kwargs["normalized_obs"],
                    )

                elif self.param.LATENT.TYPE == "DIM":
                    self.latent_model = DeepInfoMax(
                        self.obs_shape[0],
                        self.info["state_dim"],
                        self.param.LATENT,
                        img_size=self.obs_shape[-1],
                    )
                    if self.param.LATENT.DIM_LOSS == "dim2":
                        self.dim_loss = DeepInfoMaxLoss(
                            self.latent_model.local_layer_depth,
                            self.info["state_dim"],
                            self.device,
                        ).to(self.device)
                    elif self.param.LATENT.DIM_LOSS == "dim":
                        self.dim_loss = DIM_Loss(
                            self.latent_model.local_layer_depth,
                            self.info["state_dim"],
                            self.device,
                        )
                    elif self.param.LATENT.DIM_LOSS == "st-dim":
                        self.dim_loss = ST_DIM_Loss(
                            self.latent_model.local_layer_depth,
                            self.info["state_dim"],
                            self.device,
                        )
                    elif self.param.LATENT.DIM_LOSS == "jsd-dim":
                        self.dim_loss = JSD_ST_DIM_Loss(
                            self.latent_model.local_layer_depth,
                            self.info["state_dim"],
                            self.device,
                        )
                    elif self.param.LATENT.DIM_LOSS == "driml":
                        self.dim_loss = JSD_ST_DIM_Loss(
                            self.latent_model.local_layer_depth,
                            self.info["state_dim"],
                            self.device,
                        )
                    self.loss_optimizer = optim.Adam(
                        self.dim_loss.parameters(), lr=param.LATENT.LEARNING_RATE
                    )
                    self.loss_weights["deep_info_loss"] = 1
                    self.losses["deep_info_loss"] = 0

                elif self.param.LATENT.TYPE == "SLAC":
                    self.info["seq_len"] = self.param.SSM.SEQ_LEN
                    gaussian_net_args = {
                        "architecture": self.param.SSM.GAUSSIAN_NET.ARCHITECTURE,
                        "activation": self.param.SSM.GAUSSIAN_NET.ACTIVATION,
                        "batchnorm": self.param.SSM.GAUSSIAN_NET.BATCHNORM,
                        "dropout": self.param.SSM.GAUSSIAN_NET.DROPOUT,
                        "init": self.param.SSM.GAUSSIAN_NET.INIT,
                        "fixed_std": self.param.SSM.GAUSSIAN_NET.FIXED_STD,
                    }
                    self.latent_model = LatentNetwork(
                        img_channels=self.obs_shape[0],
                        state_dim=self.info["state_dim"],
                        act_dim=self.act_dim,
                        seq_len=self.param.SSM.SEQ_LEN,
                        latent1_dim=self.param.SSM.LATENT_1_DIM,
                        latent2_dim=self.param.SSM.LATENT_2_DIM,
                        normalized_obs=False,
                        img_size=self.obs_shape[-1],
                        encoder_args=self.encoder_args,
                        gaussian_args=gaussian_net_args,
                        lr=self.param.SSM.LEARNING_RATE,
                    )

                    self.loss_weights["slac_loss"] = self.param.LATENT.LOSS_WEIGHT
                    self.losses["kl_loss"] = 0
                    self.losses["reconstruction_loss"] = 0
                    self.losses["reward_log_liklihood"] = 0
                    self.losses["image_log_liklihood"] = 0
                    self.loss_weights["kl_loss"] = 1e-3
                    self.loss_weights["reconstruction_loss"] = 10
                    self.loss_weights["reward_log_liklihood"] = -1
                    self.loss_weights["image_log_liklihood"] = 0

                    self.info["method"] = "slac"

                    self.info["input_dim_actor"] = (self.info["seq_len"]) * self.info[
                        "state_dim"
                    ] + (self.info["seq_len"] - 1) * self.act_dim
                    self.info["input_dim_critic"] = (
                        self.latent_model.latent1_dim + self.latent_model.latent2_dim
                    )

                    self.info["state_type"] = "feature_action_seq"
                    self.info["sequential"] = True
                    self.info["stochastic_model"] = True
                    self.info["use_cnn"] = False

                elif self.param.LATENT.TYPE == "CURL":
                    self.image_size = self.param.CURL.IMAGE_SIZE
                    self.latent_model = CURL(
                        img_channels=self.obs_shape[0],
                        state_dim=self.info["state_dim"],
                        encoder_args=kwargs["encoder_args"],
                        lr=0.001,
                        img_size=self.image_size,
                        normalized_obs=kwargs["normalized_obs"],
                        output_type="continuous",
                        device=self.device,
                    )
                    self.loss_weights["cpc_loss"] = self.param.CURL.LOSS_WEIGHT
                    self.losses["cpc_loss"] = 0
                    self.info["contrastive"] = True
                else:
                    assert "Compression objective given, but type of compression is not implemented"

            if "FORWARD" in self.param.MODELS:
                self.forward_model = ForwardModel(
                    state_dim=self.info["state_dim"],
                    action_dim=self.act_dim,
                    param=self.param.FORWARD,
                    envtype=self.envtype,
                    device=self.device,
                )
                self.loss_weights["forward_loss"] = self.param.FORWARD.LOSS_WEIGHT
                self.losses["forward_loss"] = 0

            if "INVERSE" in self.param.MODELS:
                self.inverse_model = InverseModel(
                    state_dim=self.info["state_dim"],
                    action_dim=self.act_dim,
                    param=self.param.INVERSE,
                    envtype=self.envtype,
                    device=self.device,
                )
                self.loss_weights["inverse_loss"] = self.param.INVERSE.LOSS_WEIGHT
                self.losses["inverse_loss"] = 0

            if "REWARD" in self.param.MODELS:
                self.reward_model = RewardModel(
                    state_dim=self.info["state_dim"], n_rewards=1, device=self.device
                )
                self.loss_weights["reward_loss"] = self.param.REWARD.LOSS_WEIGHT
                self.losses["reward_loss"] = 0

            if "WORLD" in self.param.MODELS:
                assert not (
                    "LATENT" in self.param.MODELS and self.param.LATENT.TYPE == "SSM"
                ), "State Space Network cannot be used without Latent Model"
                self.info["seq_len"] = self.param.SSM.SEQ_LEN
                self.sequential = True
                self.state_space_model = WorldModel(
                    self.info["state_dim"],
                    self.act_dim,
                    self.param.RECURRENT.HIDDEN_SIZE,
                    self.param.RECURRENT.NUM_LAYERS,
                    self.device,
                )
                self.loss_weights["recurrent_loss"] = 1
                self.losses["recurrent_loss"] = 0
                self.info["state_type"] = "feature_hidden"
                self.info["input_dim_actor"] = (
                    self.info["state_dim"] + self.param.RECURRENT.HIDDEN_SIZE
                )
                self.info["input_dim_critic"] = (
                    self.info["state_dim"] + self.param.RECURRENT.HIDDEN_SIZE
                )
                self.info["sequential"] = True

            if "LSTM" in self.param.MODELS:
                assert not (
                    "LATENT" in self.param.MODELS and self.param.LATENT.TYPE == "SSM"
                ), "State Space Network cannot be used without Latent Model"
                self.info["seq_len"] = self.param.SSM.SEQ_LEN
                self.sequential = True
                self.state_space_model = WorldModel(
                    self.state_dim,
                    self.act_dim,
                    self.param.RECURRENT.HIDDEN_SIZE,
                    self.param.RECURRENT.NUM_LAYERS,
                    self.device,
                )
                self.loss_weights["recurrent_loss"] = 1
                self.losses["recurrent_loss"] = 0
                self.info["state_type"] = "feature_hidden"
                self.info["input_dim_actor"] = (
                    self.state_dim + self.param.RECURRENT.HIDDEN_SIZE
                )
                self.info["input_dim_critic"] = (
                    self.state_dim + self.param.RECURRENT.HIDDEN_SIZE
                )

            if "PRIORS" in self.param.LOSSES:
                self.loss_weights[
                    "temp_coherence_loss"
                ] = self.param.PRIORS.LOSS_WEIGHTS[0]
                self.loss_weights["causality_loss"] = self.param.PRIORS.LOSS_WEIGHTS[1]
                self.loss_weights[
                    "proportionality_loss"
                ] = self.param.PRIORS.LOSS_WEIGHTS[2]
                self.loss_weights[
                    "repeatability_loss"
                ] = self.param.PRIORS.LOSS_WEIGHTS[3]
                self.losses["temp_coherence_loss"] = 0
                self.losses["causality_loss"] = 0
                self.losses["proportionality_loss"] = 0
                self.losses["repeatability_loss"] = 0

            if "L1_REG" in self.param.LOSSES:
                self.loss_weights["l1_reg_loss"] = self.param.WEIGHTS[
                    self.param.LOSSES.index("L1_REG")
                ]
                self.losses["l1_reg_loss"] = 0

            if "L2_REG" in self.param.LOSSES:
                self.loss_weights[
                    "lposterior_hidden_pair = torch.cat([posterior, hidden_in], dim=1)2_reg_loss"
                ] = self.param.WEIGHTS[self.param.LOSSES.index("L2_REG")]
                self.losses["l2_reg_loss"] = 0

            if "RL" in self.param.LOSSES:
                self.info["rl_loss"] = True
                # self.info['state_type'] = 'state'
                # self.info['use_cnn'] = False
            else:
                if self.info["latent"]:
                    if self.info["state_type"] == "observation":
                        self.info["state_type"] = "state"
                    self.info["use_cnn"] = False
        else:
            self.info["input_dim_actor"] = self.info["state_dim"]
            self.info["input_dim_critic"] = self.info["state_dim"]
        print(self.info["state_type"])

    def get_loss_info_dict(self):
        return self.losses

    def get_info(self):
        return self.info

    def optimize(self, batch, step=0):
        obs_t = batch["obs"]  # [indices]
        obs_tp1 = batch["obs_tp1"]  # [indices]
        state_tp1 = None
        state_t = None
        srl_info = {}
        srl_info["reconstructed"] = False

        # if 'RECURRENT' in self.param.MODELS:
        if self.info["sequential"] and not self.info["stochastic_model"]:
            (batch_size, seq_len, C, H, W) = obs_tp1.shape
            obs_tp1 = obs_tp1.view(batch_size * seq_len, C, H, W)
            obs_t = obs_t.view(batch_size * seq_len, C, H, W)

        if "LATENT" in self.param.MODELS:
            # Type of autoencoder
            hidden_in = 0
            if self.param.LATENT.TYPE == "AE":
                srl_info["reconstructed"] = True
                decoded_t, state_t = self.latent_model(obs_t)
                srl_info["obs_t"] = obs_t[-1]
                if not self.normalized_obs:
                    if self.param.LATENT.USE_PREPROCESSED_IMG:
                        # preprocess images to be in [-0.5, 0.5] range
                        target_obs = preprocess_obs(obs_t)
                    else:
                        target_obs = obs_t / 255.0

                self.losses["reconstruction_loss"] = reconstruction_loss(
                    decoded_t, target_obs
                )
                srl_info["decoded_t"] = decoded_t[-1]

            elif self.param.LATENT.TYPE == "RAE":
                srl_info["reconstructed"] = True
                decoded_t, state_t = self.latent_model(obs_t)
                srl_info["obs_t"] = obs_t[-1]
                if not self.normalized_obs:
                    if self.param.LATENT.USE_PREPROCESSED_IMG:
                        # preprocess images to be in [-0.5, 0.5] range
                        target_obs = preprocess_obs(obs_t)
                    else:
                        target_obs = obs_t / 255.0
                self.losses["rae_loss"] = rae_loss(
                    decoded_t, target_obs, state_t, self.beta
                )
                srl_info["decoded_t"] = decoded_t[-1]

            elif self.param.LATENT.TYPE == "VAE":
                srl_info["reconstructed"] = True
                decoded_t, mu_t, logvar_t, state_t = self.latent_model(obs_t)
                srl_info["obs_t"] = obs_t[-1]
                if not self.normalized_obs:
                    if self.param.LATENT.USE_PREPROCESSED_IMG:
                        # preprocess images to be in [-0.5, 0.5] range
                        target_obs = preprocess_obs(obs_t)
                    else:
                        target_obs = obs_t / 255.0
                self.losses["kl_divergence"] = kl_loss(mu_t, logvar_t, beta=self.beta)
                self.losses["reconstruction_loss"] = reconstruction_loss(
                    decoded_t, target_obs
                )
                srl_info["decoded_t"] = decoded_t[-1]

            elif self.param.LATENT.TYPE == "DIM":
                srl_info["reconstructed"] = False
                y_t, M_t = self.latent_model(obs_t)
                y_tp1, M_tp1 = self.latent_model(obs_tp1)
                if self.param.LATENT.DIM_LOSS == "dim2":
                    M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
                    self.losses["deep_info_loss"] = self.dim_loss(y_t, M_t, M_prime)
                elif self.param.LATENT.DIM_LOSS == "dim":
                    self.losses["deep_info_loss"] = self.dim_loss(y_t, M_t)
                elif self.param.LATENT.DIM_LOSS == "st-dim":
                    self.losses["deep_info_loss"] = self.dim_loss(M_t, M_tp1, y_tp1)
                elif self.param.LATENT.DIM_LOSS == "jsd-dim":
                    M_t_hat = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
                    self.losses["deep_info_loss"] = self.dim_loss(
                        M_t, M_tp1, y_tp1, M_t_hat
                    )

            elif self.param.LATENT.TYPE == "CURL":
                srl_info["reconstructed"] = False
                obs_anchor, obs_pos = batch["obs"], batch["pos"]
                self.losses["cpc_loss"] = self.latent_model.get_loss(
                    obs_anchor, obs_pos
                )
                # self.latent_model.target_update()

            elif self.param.LATENT.TYPE == "ENCODER":
                srl_info["reconstructed"] = False

            elif self.param.LATENT.TYPE == "SLAC":
                if not self.normalized_obs:
                    if self.param.LATENT.USE_PREPROCESSED_IMG:
                        # preprocess images to be in [-0.5, 0.5] range
                        target_obs = preprocess_obs(obs_t)
                    else:
                        target_obs = obs_t / 255.0
                feature_seq = self.latent_model.encode(obs_t)
                action_seq = (
                    batch["act"].unsqueeze(-1)
                    if batch["act"].ndim < 3
                    else batch["act"]
                )
                reward_seq = (
                    batch["rew"].unsqueeze(-1)
                    if batch["rew"].ndim < 3
                    else batch["rew"]
                )
                done = (
                    batch["done"].unsqueeze(-1)
                    if batch["done"].ndim < 3
                    else batch["done"]
                )

                # Sample from latent variable model.
                (
                    latent1_post_mean,
                    latent1_post_std,
                    latent1_samples,
                    latent2_samples,
                ) = self.latent_model.sample_posterior(feature_seq, action_seq)
                latent1_prior_mean, latent1_prior_std = self.latent_model.sample_prior(
                    action_seq
                )

                # Calculate KL divergence loss.
                self.losses["kl_loss"] = (
                    calculate_kl_divergence(
                        latent1_post_mean,
                        latent1_post_std,
                        latent1_prior_mean,
                        latent1_prior_std,
                    )
                    .mean(dim=0)
                    .sum()
                )

                # Prediction loss of images.
                latent_state = torch.cat([latent1_samples, latent2_samples], dim=-1)
                decoded_obs = self.latent_model.decode(latent_state)
                decoded_obs_seq_std = torch.ones_like(decoded_obs).mul_(1.0)

                obs_noise = (target_obs - decoded_obs) / (decoded_obs + 1e-8)
                log_likelihood_ = (
                    -0.5 * obs_noise.pow(2) - decoded_obs.log()
                ) - 0.5 * math.log(2 * math.pi)
                self.losses["image_log_liklihood"] = log_likelihood_.mean(dim=0).sum()
                self.losses["reconstruction_loss"] = reconstruction_loss(
                    decoded_obs, target_obs
                )

                # Prediction loss of rewards.
                x = torch.cat(
                    [latent_state[:, :-1], action_seq, latent_state[:, 1:]], dim=-1
                )
                B, S, X = x.shape
                reward_mean, reward_std = self.latent_model.reward_predictor(
                    x.view(B * S, X)
                )
                reward_mean = reward_mean.view(B, S, 1)
                reward_std = reward_std.view(B, S, 1)

                reward_noise = (reward_seq - reward_mean) / (reward_std + 1e-8)
                log_likelihood_reward = (
                    -0.5 * reward_noise.pow(2) - reward_std.log()
                ) - 0.5 * math.log(2 * math.pi)
                self.losses["reward_log_liklihood"] = (
                    log_likelihood_reward.mul_(1 - done).mean(dim=0).sum()
                )

                srl_info["reconstructed"] = True
                srl_info["obs_t"] = obs_t[-1]
                srl_info["decoded_t"] = decoded_obs[-1]

        if "FORWARD" in self.param.MODELS:
            state_t = self.latent_model.get_state(obs_t)
            state_tp1 = self.latent_model.get_state(obs_tp1)
            state_tp1_pred = self.forward_model(state_t, batch["act"])
            self.losses["forward_loss"] = reconstruction_loss(state_tp1_pred, state_tp1)

        if "INVERSE" in self.param.MODELS:
            state_tp1 = self.latent_model.get_state(obs_tp1)
            act_pred = self.inverse_model(state_t, state_tp1)
            act = batch["act"].unsqueeze(-1) if batch["act"].ndim < 2 else batch["act"]
            self.losses["inverse_loss"] = reconstruction_loss(act_pred, act)

        if "REWARD" in self.param.MODELS:
            state_tp1 = self.latent_model.get_state(obs_tp1)
            rew_pred = self.reward_model(state_t, state_tp1)
            rew = batch["rew"]
            if rew_pred.ndim > rew.ndim:
                rew = rew.unsqueeze(-1)
            self.losses["reward_loss"] = reconstruction_loss(rew_pred, rew)

        if "WORLD" in self.param.MODELS:
            if step < self.param.SSM.ENCODER_PRETRAINING_STEPS:
                self.losses["recurrent_loss"] = torch.tensor(0.0).to(self.device)
            else:
                self.losses["reconstruction_loss"] = torch.tensor(0.0).to(self.device)
                self.loss_weights["reconstruction_loss"] = 0
                self.train_latent = False
                with torch.no_grad():
                    state_t = self.latent_model.get_state(obs_t)
                state_t = state_t.view(batch_size, seq_len, -1)
                state_tp1_true = state_t[:, 1:, :]
                state_t = state_t[:, :-1, :]
                state_tp1_pred, _ = self.state_space_model(
                    state_t.detach(), batch["act"].unsqueeze(-1)[:, :-1, :]
                )
                self.losses["recurrent_loss"] = l1_loss(
                    state_tp1_pred, state_tp1_true.detach()
                )

        if "PRIORS" in self.param.LOSSES:
            if not state_t:
                state_t = self.latent_model.get_state(obs_t, grad=True)
            if not state_tp1:
                state_tp1 = self.latent_model.get_state(obs_tp1, grad=True)
            test = robotic_prior_loss(state_t, state_tp1, batch["act"], batch["rew"])
            self.losses.update(
                robotic_prior_loss(state_t, state_tp1, batch["act"], batch["rew"])
            )

        if "L1_REG" in self.param.LOSSES:
            self.losses["l1_reg_loss"] = 0

        if "L2_REG" in self.param.LOSSES:
            self.losses["l2_reg_loss"] = 0

        srl_losses = {}
        loss = 0
        for key, val in self.losses.items():
            if isNaN(val) and key == "image_log_liklihood":
                val = torch.zeros_like(val)
            loss = loss + val * self.loss_weights[key]
            srl_losses[key] = val.detach().cpu().numpy() * self.loss_weights[key]

        if not srl_losses:
            return srl_losses, srl_info

        if "LATENT" in self.param.MODELS and self.train_latent:
            self.latent_model.optimizer.zero_grad()
        if "FORWARD" in self.param.MODELS:
            self.forward_model.optimizer.zero_grad()
        if "INVERSE" in self.param.MODELS:
            self.inverse_model.optimizer.zero_grad()
        if "REWARD" in self.param.MODELS:
            self.reward_model.optimizer.zero_grad()
        if "RECURRENT" in self.param.MODELS:
            self.state_space_model.optimizer.zero_grad()
        if self.param.LATENT.TYPE == "DIM":
            self.loss_optimizer.zero_grad()

        loss.backward()

        if self.grad_clip:
            for p in self.modules():
                torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)

        if self.param.LATENT.TYPE == "DIM":
            self.loss_optimizer.step()
        if "LATENT" in self.param.MODELS and self.train_latent:
            self.latent_model.optimizer.step()
            if (
                self.param.LATENT.TYPE == "CURL"
                and step % 2 == 0
                and self.do_target_update
            ):
                self.latent_model.target_update()
        if "FORWARD" in self.param.MODELS:
            self.forward_model.optimizer.step()
        if "INVERSE" in self.param.MODELS:
            self.inverse_model.optimizer.step()
        if "REWARD" in self.param.MODELS:
            self.reward_model.optimizer.step()
        if "RECURRENT" in self.param.MODELS:
            self.state_space_model.optimizer.step()

        return srl_losses, srl_info

    def convert_batch(self, batch):
        """

        """
        grad = self.info["rl_loss"]
        obs_t = batch["obs"]
        obs_tp1 = batch["obs_tp1"]
        state_tp1 = None

        if self.info["sequential"] and not self.info["stochastic_model"]:
            (batch_size, seq_len, C, H, W) = obs_tp1.shape
            obs_tp1 = obs_tp1.view(batch_size * seq_len, C, H, W)
            obs_t = obs_t.view(batch_size * seq_len, C, H, W)
            if self.info["contrastive"]:
                if obs_t.shape[-1] != self.image_size:
                    obs_t = center_crop_image(obs_t, self.image_size)
                if obs_tp1.shape[-1] != self.image_size:
                    obs_tp1 = center_crop_image(obs_tp1, self.image_size)
        else:
            if self.info["contrastive"]:
                if obs_t.shape[-1] != self.image_size:
                    obs_t = center_crop_image(obs_t, self.image_size)
                if obs_tp1.shape[-1] != self.image_size:
                    obs_tp1 = center_crop_image(obs_tp1, self.image_size)

        if (
            self.info["use_srl"]
            and self.info["latent"]
            and not self.info["state_type"] == "observation"
        ):
            hidden_in = 0

            if self.info["state_type"] == "feature_hidden":
                features = self.latent_model.get_state(obs_t, detach=False).view(
                    batch_size, seq_len, -1
                )
                hidden = self.state_space_model.infer(
                    features, batch["act"].unsqueeze(-1)
                )[0].transpose(1, 0)
                last_feature = features[:, -1]
                hidden = hidden.view(1, -1)
                states = torch.cat([features, hidden], dim=-1)  # .unsqueeze(0)
                # if states.ndim <3: states = state.unsqueeze(0)
                batch["act"] = batch["act"][:, -1]
                batch["rew"] = batch["rew"][:, -1]
                batch["done"] = batch["done"][:, -1]
                batch["actor_state_t"], batch["actor_state_tp1"] = states, states
                batch["critic_state_t"], batch["critic_state_tp1"] = (
                    batch["actor_state_t"],
                    batch["actor_state_tp1"],
                )
                batch["critic_target_state_t"], batch["critic_target_state_tp1"] = (
                    batch["critic_state_t"],
                    batch["critic_state_tp1"],
                )

            elif self.info["state_type"] == "feature_action_seq":
                action_seq = (
                    batch["act"].unsqueeze(-1)
                    if batch["act"].ndim < 3
                    else batch["act"]
                )
                with torch.no_grad():
                    # f(1:t+1)
                    features_seq = self.latent_model.get_state(obs_t, detach=True)
                    # z(1:t+1)
                    (
                        _,
                        _,
                        latent1_samples,
                        latent2_samples,
                    ) = self.latent_model.sample_posterior(features_seq, action_seq)
                # z(t), z(t+1)
                latents_seq = torch.cat([latent1_samples, latent2_samples], dim=-1)
                # input policy
                latent_t = latents_seq[:, -2]
                latent_tp1 = latents_seq[:, -1]
                batch["critic_state_t"], batch["critic_state_tp1"] = (
                    latent_t,
                    latent_tp1,
                )
                batch["critic_target_state_t"], batch["critic_target_state_tp1"] = (
                    batch["critic_state_t"],
                    batch["critic_state_tp1"],
                )
                # a(t)
                actions = action_seq[:, -1]
                # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
                feature_actions, next_feature_actions = create_feature_actions(
                    features_seq, action_seq
                )
                # input policy
                # sequence of features
                batch["actor_state_t"], batch["actor_state_tp1"] = (
                    feature_actions,
                    next_feature_actions,
                )
                batch["act"], batch["rew"], batch["done"] = (
                    actions,
                    batch["rew"][:, -1],
                    batch["done"][:, -1],
                )
            elif self.info["data_augmentation"]:
                obs_t_aug = batch["obs_aug"]
                obs_tp1_aug = batch["obs_tp1_aug"]
                batch["critic_state_t"], batch["critic_state_tp1"] = (
                    self.latent_model.get_state(obs_t),
                    self.latent_model.get_state(obs_tp1),
                )
                batch["critic_state_t_aug"], batch["critic_state_tp1_aug"] = (
                    self.latent_model.get_state(obs_t_aug),
                    self.latent_model.get_state(obs_tp1_aug),
                )

                batch["actor_state_t_aug"], batch["actor_state_tp1_aug"] = (
                    self.latent_model.get_state(obs_t_aug, detach=True),
                    self.latent_model.get_state(obs_tp1_aug, detach=True),
                )
                batch["actor_state_t"], batch["actor_state_tp1"] = (
                    self.latent_model.get_state(obs_t, detach=True),
                    self.latent_model.get_state(obs_tp1, detach=True),
                )
            else:
                batch["critic_state_t"], batch["critic_state_tp1"] = (
                    self.latent_model.get_state(obs_t),
                    self.latent_model.get_state(obs_tp1),
                )
                batch["actor_state_t"], batch["actor_state_tp1"] = (
                    self.latent_model.get_state(obs_t, detach=True),
                    self.latent_model.get_state(obs_tp1, detach=True),
                )

        else:
            batch["critic_state_t"], batch["critic_state_tp1"] = obs_t, obs_tp1
            batch["actor_state_t"], batch["actor_state_tp1"] = obs_t, obs_tp1
            if self.info["data_augmentation"]:
                obs_t_aug = batch["obs_aug"]
                obs_tp1_aug = batch["obs_tp1_aug"]
                batch["critic_state_t_aug"], batch["critic_state_tp1_aug"] = (
                    obs_t_aug,
                    obs_tp1_aug,
                )
                batch["actor_state_t_aug"], batch["actor_state_tp1_aug"] = (
                    obs_t_aug,
                    obs_tp1_aug,
                )

        if not self.info["rl_loss"]:
            batch["critic_state_t"], batch["critic_state_tp1"] = (
                batch["critic_state_t"].detach(),
                batch["critic_state_tp1"].detach(),
            )
            batch["actor_state_t"], batch["actor_state_tp1"] = (
                batch["actor_state_t"].detach(),
                batch["actor_state_tp1"].detach(),
            )
        return batch

    def get_state(self, obs):
        with torch.no_grad():
            if not self.info["latent"]:
                state = torch.FloatTensor(obs).to(self.device)
                if len(state.shape) == 3:
                    state = state.unsqueeze(0)
                features = 0
            elif self.info["state_type"] == "feature_action_seq":
                obs, action = self.deque_to_seq(obs)
                with torch.no_grad():
                    # only for slac
                    feature = self.latent_model.get_state(obs)
                state = torch.cat([feature.view(1, -1), action.view(1, -1)], dim=-1)
                features = 0
            elif self.info["state_type"] == "feature_hidden":
                if self.info["contrastive"]:
                    if obs.shape[-1] != self.image_size:
                        obs = center_crop_image(obs, self.image_size)
                obs = torch.FloatTensor(obs).to(self.device)
                if len(obs.shape) == 3:
                    obs = obs.unsqueeze(0)
                features = self.latent_model.get_state(obs, detach=False)
                hidden = self.state_space_model.hidden[0].squeeze(0)[0]
                state = torch.cat([features, hidden], dim=1)
            elif self.info["state_type"] == "state":
                if self.info["contrastive"]:
                    if obs.shape[-1] != self.image_size:
                        obs = center_crop_image(obs, self.image_size)
                obs = torch.FloatTensor(obs).to(self.device)
                if len(obs.shape) == 3:
                    obs = obs.unsqueeze(0)
                state = self.latent_model.get_state(obs, detach=False)
                features = 0
            elif self.info["state_type"] == "observation":
                if self.info["contrastive"]:
                    if obs.shape[-1] != self.image_size:
                        obs = center_crop_image(obs, self.image_size)
                obs = torch.FloatTensor(obs).to(self.device)
                if len(obs.shape) == 3:
                    obs = obs.unsqueeze(0)
                state = obs
                features = 0
        return state, features

    def deque_to_seq(self, obs):
        obs_deque, act_deque = obs[0], obs[1]
        obs = np.array(obs_deque, dtype=np.float32)
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        action = np.array(act_deque, dtype=np.float32)
        action = torch.FloatTensor(action).to(self.device).unsqueeze(0)
        return obs, action

    def target_update(self):
        tau = 0.005
        with torch.no_grad():
            for p, p_targ in zip(
                self.latent_model.parameters(), self.latent_model_target.parameters()
            ):
                p_targ.data.copy_(tau * p.data + (1 - tau) * p_targ.data)


def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top : top + new_h, left : left + new_w]
    return image


def create_feature_actions(features_seq, actions_seq):
    N = features_seq.size(0)

    # sequence of features
    f = features_seq[:, :-1].view(N, -1)
    n_f = features_seq[:, 1:].view(N, -1)
    # sequence of actions
    a = actions_seq[:, :-1].view(N, -1)
    n_a = actions_seq[:, 1:].view(N, -1)

    # feature_actions
    fa = torch.cat([f, a], dim=-1)
    n_fa = torch.cat([n_f, n_a], dim=-1)

    return fa, n_fa
