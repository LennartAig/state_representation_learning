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
        self.clip_ratio = kwargs['clip_ratio']
        self.entropy_coef = kwargs['entropy_coef']
        self.value_coef = kwargs['value_coef']
        self.use_grad_clipping = kwargs['use_grad_clipping']
        self.max_grad_norm = kwargs ['max_grad_norm']
        self.clipping_range = kwargs['clipping_range']
        self.action_limit = kwargs['action_limit']
        self.use_value_clipping = kwargs['use_value_clipping']
        kwargs['squashed'] = False
        kwargs['fixed_std'] = True
        self.squashed = kwargs['squashed']
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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)         
        self.actor.train()
        self.critic.train()
       
    def step(self, obs, deterministic = False):
        if len(obs.shape) == 3 or obs.ndim == 1: obs = obs.unsqueeze(0)
        with torch.no_grad():
            value = self.critic(obs)
            pi,_ = self.actor(obs)
            action = pi.mode() if deterministic else pi.sample()
            log_prob = pi.log_probs(action)
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
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        test = clip_adv.mean()
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent.mean().item(), cf=clipfrac)

        return loss_pi, ent, pi_info
    
    def compute_loss_v(self, obs, ret, old_val):
        val = self.critic(obs)
        test = (val- ret)**2
        if self.use_value_clipping:
            val = old_val + torch.clamp(val - old_val, -self.clipping_range, self.clipping_range)
        return 0.5*((val- ret)**2).mean()
    
    def optimize(self, batch, latent_optimizer = None):
        # Train policy with multiple steps of gradient descent
        pi_l_old, _, pi_info_old = self.compute_loss_pi(batch['actor_state_t'], batch['act'], batch['adv'], batch['logp'])
        v_l_old = self.compute_loss_v(batch['critic_state_t'], batch['ret'], batch['val'])
        loss_pi, entropy, pi_info = self.compute_loss_pi(batch['actor_state_t'], batch['act'], batch['adv'], batch['logp'])
        loss_entropy= -torch.mean(entropy)
        loss_v = self.compute_loss_v(batch['critic_state_t'], batch['ret'], batch['val'])

        loss = loss_pi + self.value_coef * loss_v + self.entropy_coef * loss_entropy
        if self.use_grad_clipping:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        if latent_optimizer is not None: latent_optimizer.zero_grad()    
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        self.actor_optimizer.step()
        if latent_optimizer is not None: latent_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        return dict(LossPi=loss_pi.item(), LossV=loss_v.item(),
                    KL=kl, Entropy=ent, ClipFrac=cf,
                    DeltaLossPi=(loss_pi.item() - pi_l_old.item()),
                    DeltaLossV=(loss_v.item() - v_l_old.item()))