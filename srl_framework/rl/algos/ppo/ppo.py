import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy

from srl_framework.rl.algos.base import RLAgent
from srl_framework.rl.algos.ppo.core import ValueFunction, Policy, weight_init, MLPGaussianActor, MLPCritic


class PPOAgent(nn.Module):
    """
    Vanilla Policy Gradient with value function as baseline and 
    GAE-Lambda for advantage estimation.
    """
    def __init__(self, encoder=None, **kwargs):
        super(PPOAgent, self).__init__()
        self.clip_ratio = kwargs['clip_ratio']
        self.policy_epochs = kwargs['policy_epochs']
        self.critic_epochs = kwargs['critic_epochs']
        self.early_stopping = kwargs['early_stopping']
        self.max_kl_div = kwargs['max_kl_div']
        kwargs['squashed'] = False
        kwargs['fixed_std'] = True
        self.squashed = kwargs['squashed']
        self.action_limit = kwargs['action_limit']
        self.drq = kwargs['data_regularization']
        if kwargs['use_cnn']:
            kwargs['input_dim_actor'] = kwargs['feature_dim']
            kwargs['input_dim_critic'] = kwargs['feature_dim']
        
        self.critic = ValueFunction(encoder = encoder, **kwargs).to(kwargs['device'])
        print('Critic')
        print(self.critic)
        actor_encoder = deepcopy(self.critic.encoder) if self.critic.use_encoder else None
        self.actor = Policy(encoder = actor_encoder, **kwargs).to(kwargs['device'])
        print('Actor')
        print(self.actor)

        self.to(kwargs["device"])
        self.critic.apply(weight_init)
        self.actor.apply(weight_init)
        if self.actor.use_encoder: self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        """
        self.actor=MLPGaussianActor(obs_dim=kwargs['input_dim_actor'],act_dim=kwargs['action_dim'],hidden_sizes=(64,64),activation=nn.ReLU)
        self.critic=MLPCritic(obs_dim=kwargs['input_dim_actor'],hidden_sizes=(64,64),activation=nn.ReLU)
        """
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.actor.train()
        self.critic.train()

    def step(self, obs, deterministic = False):
        """
        TODO: Pay attention on state dimension [batch_size, state_size] or [state_size]
        """
        if len(obs.shape) == 3 or obs.ndim == 1: obs = obs.unsqueeze(0)
        with torch.no_grad():
            value = self.critic(obs)
            pi,_ = self.actor(obs)
            action = pi.mode() if deterministic else pi.sample()
            log_prob = pi.log_probs(action)
            #action = pi.mean if deterministic else pi.sample()
            #log_prob = self.actor._log_prob_from_distribution(pi, action)
            if self.squashed:
                log_prob -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1, keepdim=True)
                action = torch.tanh(action) * self.action_limit
            
        return action.cpu().data.numpy().flatten(), log_prob.cpu().data.numpy().flatten(), value.cpu().data.numpy().flatten()
    
    def get_loss_info_dict(self):
        return dict(LossPi=0, LossV=0,
                    KL=0, Entropy=0, ClipFrac=0,
                    DeltaLossPi=0,
                    DeltaLossV=0)

    def compute_loss_pi(self, obs, act, adv, logp_old):
        # Policy loss (TODO: ACT DIM!!!!!!)
        pi, logp = self.actor(obs, act)
        if logp.ndim == 2:
            logp = logp.squeeze(-1)
        assert logp.ndim == 1
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 0.8, 1.2) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info
    
    def compute_loss_v(self, obs, ret):
        val = self.critic(obs)
        return ((val - ret)**2).mean()
    
    def optimize(self, batch, latent_optimizer = None):
        pi_l_old, pi_info_old = self.compute_loss_pi(batch['actor_state_t'], batch['act'], batch['adv'], batch['logp'])
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(batch['critic_state_t'], batch['ret']).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.policy_epochs):
            self.actor_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(batch['actor_state_t'], batch['act'], batch['adv'], batch['logp'])
            kl = pi_info['kl']
            if self.early_stopping and pi_info['kl'] > 1.5 * self.max_kl_div:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            self.actor_optimizer.step()

        # Value function learning
        for i in range(self.critic_epochs):
            self.critic_optimizer.zero_grad()
            loss_v = self.compute_loss_v(batch['critic_state_t'], batch['ret'])
            loss_v.backward()
            self.critic_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        return dict(LossPi=pi_l_old, LossV=v_l_old,
                    KL=kl, Entropy=ent, ClipFrac=cf,
                    DeltaLossPi=(loss_pi.item() - pi_l_old),
                    DeltaLossV=(loss_v.item() - v_l_old))