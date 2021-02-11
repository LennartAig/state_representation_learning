from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from srl_framework.rl.algos.sac.core import (
    SquashedGaussianPolicy,
    DoubleActionValueFunction,
    weight_init,
)


class SACAgent(nn.Module):
    def __init__(self, encoder=None, polyak=0.995, gamma=0.99, **kwargs):
        super(SACAgent, self).__init__()
        self.polyak = polyak
        self.gamma = gamma
        self.drq = kwargs["data_regularization"]
        if kwargs["use_cnn"]:
            kwargs["input_dim_actor"] = kwargs["feature_dim"]
            kwargs["input_dim_critic"] = kwargs["feature_dim"]

        self.critic = DoubleActionValueFunction(encoder=encoder, **kwargs).to(
            kwargs["device"]
        )
        print("Critic")
        print(self.critic)
        actor_encoder = (
            deepcopy(self.critic.encoder) if self.critic.use_encoder else None
        )
        self.actor = SquashedGaussianPolicy(encoder=actor_encoder, **kwargs).to(
            kwargs["device"]
        )
        print("Actor")
        print(self.actor)
        if kwargs["target_encoder"]:
            self.critic_target = DoubleActionValueFunction(
                encoder=kwargs["target_encoder"], **kwargs
            ).to(kwargs["device"])
            self.critic_target.load_state_dict(self.critic.state_dict())
        else:
            self.critic_target = deepcopy(self.critic)
            # self.critic_target = DoubleActionValueFunction(encoder = encoder, **kwargs).to(kwargs['device'])
            # self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(0.01)).to(kwargs["device"])
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(kwargs["action_dim"])

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.to(kwargs["device"])

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=1e-3, betas=(0.9, 0.999)
        )
        self.critic.apply(weight_init)
        self.actor.apply(weight_init)
        if self.actor.use_encoder:
            self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        self.actor.train()
        self.critic.train()
        self.critic_target.train()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_loss_info_dict(self):
        return dict(Q1Vals=0, Q2Vals=0, LossQ=0, LogPi=0, LossPi=0)

    def target_update(self):
        with torch.no_grad():

            for p, p_targ in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def step(self, obs, deterministic=False):
        if len(obs.shape) == 3 or obs.ndim == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(obs, deterministic=deterministic)
        return action.cpu().data.numpy().flatten(), 0, 0

    def compute_loss_q(self, batch):
        act, rew, not_done = batch["act"], batch["rew"], 1 - batch["done"]
        if not_done.ndim == 1:
            not_done = not_done.unsqueeze(-1)
            rew = rew.unsqueeze(-1)
        actor_state_tp1 = batch["actor_state_tp1"]
        critic_state_t = batch["critic_state_t"]
        critic_target_state_tp1 = batch["critic_state_tp1"]
        if self.drq:
            actor_state_tp1_aug = batch["actor_state_tp1_aug"]
            critic_state_t_aug = batch["critic_state_t_aug"]
            critic_target_state_tp1_aug = batch["critic_target_state_tp1_aug"]
        if len(act.shape) <= 1:
            act = act.unsqueeze(-1)

        with torch.no_grad():
            # Target actions come from *current* policy
            new_act_tp1, log_p_tp1 = self.actor(actor_state_tp1)
            # Target Q-Values
            q1_target_val, q2_target_val = self.critic_target(
                critic_target_state_tp1, new_act_tp1
            )
            if log_p_tp1.ndim > q1_target_val.ndim:
                log_p_tp1 = log_p_tp1.squeeze(-1)
            q_target_val = (
                torch.min(q1_target_val, q2_target_val)
                - self.alpha.detach() * log_p_tp1
            )

            bellmann_backup = rew + (self.gamma * not_done * q_target_val)
            if self.drq:
                # Target actions come from *current* policy
                new_act_tp1_aug, log_p_tp1_aug = self.actor(actor_state_tp1_aug)
                # Target Q-Values
                q1_target_val_aug, q2_target_val_aug = self.critic_target(
                    critic_target_state_tp1_aug, new_act_tp1_aug
                )
                q_target_val_aug = (
                    torch.min(q1_target_val_aug, q2_target_val_aug)
                    - self.alpha.detach() * log_p_tp1
                )
                bellmann_backup_aug = rew + (self.gamma * not_done * q_target_val_aug)

                bellmann_backup = (bellmann_backup + bellmann_backup_aug) / 2

        q1_val, q2_val = self.critic(critic_state_t, act)
        # MSE loss against Bellman backup
        loss_q1 = F.mse_loss(q1_val, bellmann_backup)
        loss_q2 = F.mse_loss(q2_val, bellmann_backup)
        critic_loss = loss_q1 + loss_q2
        if self.drq:
            q1_val_aug, q2_val_aug = self.critic(critic_state_t_aug, act)
            # MSE loss against Bellman backup
            loss_q1_aug = F.mse_loss(q1_val_aug, bellmann_backup)
            loss_q2_aug = F.mse_loss(q2_val_aug, bellmann_backup)
            critic_loss_aug = loss_q1_aug + loss_q2_aug

            critic_loss += critic_loss_aug
        q_info = dict(
            Q1Vals=q1_val.detach().cpu().numpy(),
            Q2Vals=q2_val.detach().cpu().numpy(),
            LossQ=critic_loss.detach().cpu().numpy(),
        )

        return critic_loss, q_info

    def compute_loss_pi(self, batch, detach=True):
        actor_state_t = batch["actor_state_t"]
        critic_state_t = batch["critic_state_t"]
        new_act, log_p = self.actor(actor_state_t, detach_encoder=detach)
        q1_val, q2_val = self.critic(critic_state_t, new_act, detach_encoder=detach)
        if log_p.ndim > q1_val.ndim:
            log_p = log_p.squeeze(-1)
        q_val = torch.min(q1_val, q2_val)
        actor_loss = (self.alpha.detach() * log_p - q_val).mean()
        alpha_loss = (self.alpha * (-log_p - self.target_entropy).detach()).mean()

        pi_info = dict(
            LogPi=log_p.detach().cpu().numpy(), LossPi=actor_loss.detach().cpu().numpy()
        )

        return actor_loss, pi_info, alpha_loss

    def optimize(self, batch, steps=0, latent_optimizer=None):

        loss_q, q_info = self.compute_loss_q(batch)
        if latent_optimizer:
            latent_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss_q.backward()
        self.critic_optimizer.step()
        if latent_optimizer:
            latent_optimizer.step()
        loss_pi, pi_info, alpha_loss = self.compute_loss_pi(batch, detach=True)

        self.actor_optimizer.zero_grad()
        loss_pi.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if steps % 2 == 0:
            self.target_update()
        return {**pi_info, **q_info}
