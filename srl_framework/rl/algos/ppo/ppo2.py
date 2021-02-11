import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy

from srl_framework.rl.algos.base import RLAgent
from srl_framework.rl.algos.ppo.core import ValueFunction, Policy, weight_init


class PPOAgent2(nn.Module):
    """
    Vanilla Policy Gradient with value function as baseline and 
    GAE-Lambda for advantage estimation.
    """

    def __init__(self, encoder=None, **kwargs):
        super(PPOAgent2, self).__init__()
        self.clip_ratio = kwargs["clip_ratio"]
        self.entropy_coef = kwargs["entropy_coef"]
        self.value_coef = kwargs["value_coef"]
        self.use_grad_clipping = kwargs["use_grad_clipping"]
        self.max_grad_norm = kwargs["max_grad_norm"]
        self.clipping_range = kwargs["clipping_range"]
        self.action_limit = kwargs["action_limit"]
        self.use_value_clipping = kwargs["use_value_clipping"]
        self.early_stopping = kwargs["early_stopping"]

        kwargs["squashed"] = False
        kwargs["fixed_std"] = True
        self.squashed = kwargs["squashed"]
        self.drq = kwargs["data_regularization"]
        self.scheduler = kwargs["lr_scheduler"]
        self.advantage_norm = kwargs["advantage_normalization"]
        self.cutoff_coef = kwargs["cutoff_coef"]
        self.max_kl_div = kwargs["max_kl_div"]
        if kwargs["use_cnn"]:
            kwargs["input_dim_actor"] = kwargs["feature_dim"]
            kwargs["input_dim_critic"] = kwargs["feature_dim"]

        self.critic = ValueFunction(encoder=encoder, **kwargs).to(kwargs["device"])
        print("Critic")
        print(self.critic)
        actor_encoder = (
            deepcopy(self.critic.encoder) if self.critic.use_encoder else None
        )
        self.actor = Policy(encoder=actor_encoder, **kwargs).to(kwargs["device"])
        print("Actor")
        print(self.actor)

        self.to(kwargs["device"])
        self.critic.apply(weight_init)
        self.actor.apply(weight_init)
        if self.actor.use_encoder:
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        if kwargs["lr_scheduler"]:
            schedule = lambda epoch: 1 - epoch / kwargs["schedule_param"]
            self.actor_scheduler = optim.lr_scheduler.LambdaLR(
                self.actor_optimizer, schedule
            )
            self.critic_scheduler = optim.lr_scheduler.LambdaLR(
                self.critic_optimizer, schedule
            )
        self.actor.train()
        self.critic.train()

    def step(self, obs, deterministic=False):
        if len(obs.shape) == 3 or obs.ndim == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            value = self.critic(obs)
            pi, _ = self.actor(obs)
            action = pi.mode() if deterministic else pi.sample()
            log_prob = pi.log_probs(action)
            if self.squashed:
                log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(
                    axis=1, keepdim=True
                )
                action = torch.tanh(action) * self.action_limit

        return (
            action.cpu().data.numpy().flatten(),
            log_prob.cpu().data.numpy().flatten(),
            value.cpu().data.numpy().flatten(),
        )

    def get_loss_info_dict(self):
        return dict(LossPi=0, LossV=0, KL=0, Entropy=0, ClipFrac=0)

    def compute_loss_pi(self, obs, act, adv, logp_old):
        if self.advantage_norm:
            adv_std = adv.std() + 1e-5
            adv_mean = adv.mean()
            adv_norm = (adv - adv_mean) / adv_std
            adv = adv_norm
        pi, logp = self.actor(obs, act, detach_encoder=False)
        if logp.ndim == 2:
            logp = logp.squeeze(-1)
        assert logp.ndim == 1
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        kl_div = (logp_old - logp).mean()
        loss_pi += (
            self.cutoff_coef
            * (kl_div > 2 * self.max_kl_div)
            * (kl_div - self.max_kl_div) ** 2
        )

        # Useful extra info
        approx_kl = kl_div.item()
        ent = pi.entropy()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent.mean().item(), cf=clipfrac)

        return loss_pi, ent, pi_info

    def compute_loss_v(self, obs, ret, old_val):
        val = self.critic(obs)
        test = (val - ret) ** 2
        if self.use_value_clipping:
            val = old_val + torch.clamp(
                val - old_val, -self.clipping_range, self.clipping_range
            )
        return F.mse_loss(ret, val)

    def optimize(self, batch, latent_optimizer=None, num_minibatch=0):
        loss_v = self.compute_loss_v(
            batch["critic_state_t"], batch["ret_gae"], batch["val"]
        )
        loss_v = self.value_coef * loss_v

        self.critic_optimizer.zero_grad()
        loss_v.backward()
        self.critic_optimizer.step()

        loss_pi, entropy, pi_info = self.compute_loss_pi(
            batch["actor_state_t"], batch["act"], batch["adv"], batch["logp"]
        )
        loss_entropy = -torch.mean(entropy)
        loss_pi = loss_pi + self.entropy_coef * loss_entropy
        kl, ent, cf = pi_info["kl"], pi_info["ent"], pi_info["cf"]

        self.actor_optimizer.zero_grad()
        loss_pi.backward()
        self.actor_optimizer.step()

        if self.scheduler:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        if latent_optimizer is not None:
            latent_optimizer.step()

        return dict(
            LossPi=loss_pi.item(), LossV=loss_v.item(), KL=kl, Entropy=ent, ClipFrac=cf
        )
