import itertools
from copy import deepcopy

import torch
import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from srl_framework.rl.utils import vector_to_parameters, parameters_to_vector

from srl_framework.rl.algos.base import RLAgent
from srl_framework.rl.algos.trpo.core import ValueFunction, Policy, weight_init

EPS = 1e-8


class TRPOAgent(nn.Module):


    def __init__(self, encoder=None, **kwargs):
        super(TRPOAgent, self).__init__()
        self.entropy_coef = kwargs["entropy_coef"]
        self.cg_damping = kwargs["cg_damping"]
        self.cg_steps = kwargs["cg_steps"]
        self.value_epochs = kwargs["value_epochs"]
        self.num_backtrack = kwargs["num_backtrack"]
        self.delta = kwargs["delta"]
        self.alpha = kwargs["alpha"]
        kwargs["squashed"] = False
        kwargs["fixed_std"] = True
        self.squashed = kwargs["squashed"]
        self.drq = kwargs["data_regularization"]
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
        self.actor.train()
        self.critic.train()

    def policy_gradient(self, states, actions, advantages, log_probs):
        _, log_probs_new = self.actor(states, actions)
        if log_probs_new.ndim == 2:
            log_probs_new = log_probs_new.squeeze(-1)
        assert log_probs_new.ndim == log_probs.ndim
        pg_objective = (log_probs_new * advantages).mean()
        pg_objective -= self.entropy_coef * log_probs.mean()
        parameters = self.actor.parameters()
        return parameters_to_vector(torch.autograd.grad(pg_objective, parameters))

    def natural_gradient(self, pg, states):
        def Hx(x):
            """ Computes the Hessian-Vector product for the KL-Divergance:
            Needs to be definded here. all parameteres and models are given 
            before an update and then teh kl div is looked at compared to this."""
            parameters = self.actor.parameters()

            d_kl = self.get_kl(self.actor, states)
            grads = torch.autograd.grad(d_kl, parameters, create_graph=True)
            grads = parameters_to_vector(grads)
            Jx = torch.sum(grads * x)
            parameters = self.actor.parameters()
            Hx = torch.autograd.grad(Jx, parameters)
            Hx = parameters_to_vector(Hx)
            return Hx + self.cg_damping * x

        stepdir = conjugate_gradient(Hx, pg, self.cg_steps)
        stepsize = (2 * self.delta) / torch.dot(stepdir, Hx(stepdir))
        return torch.sqrt(stepsize) * stepdir

    def get_kl(self, actor_model, states):
        """ Computes the KL-Divergance between the current policy and the model passed """
        with torch.no_grad():
            p_old, _ = self.actor(states)
        p_new, _ = actor_model(states)
        d_kl = kl_divergence(p_old, p_new).sum(dim=-1, keepdim=True).mean()
        return d_kl

    def optimize_actor(self, new_parameters):
        parameters = self.actor.parameters()
        vector_to_parameters(new_parameters, parameters)

    def get_loss_info_dict(self):
        return dict(critic_loss=0, explained_variance=0, entropy=0, kl=0, pg_norm=0)

    def optimize(self, batch, latent_model_optimizer=None):
        # advatage normalization TODO ?
        states, act, adv, log_probs, ret = (
            batch["critic_state_t"],
            batch["act"],
            batch["adv"],
            batch["logp"],
            batch["ret_gae"],
        )
        # Compute Advantages
        for _ in range(self.value_epochs):
            # Update Critic
            self.critic_optimizer.zero_grad()
            values = self.critic(states)
            critic_loss = F.mse_loss(values, ret)
            critic_loss.backward()
            self.critic_optimizer.step()

        # Update Actor
        with torch.no_grad():
            _, old_log_probs = self.actor(states, act)
        if old_log_probs.ndim == 2:
            old_log_probs = old_log_probs.squeeze(-1)
        assert old_log_probs.ndim == log_probs.ndim
        pg = self.policy_gradient(states, act, adv, log_probs)
        npg = self.natural_gradient(pg, states)
        parameters, pg_norm = self.linesearch(npg, pg, states)
        self.optimize_actor(parameters)
        pi_dist, log_probs = self.actor(states, act)
        if log_probs.ndim == 2:
            log_probs = log_probs.squeeze(-1)
        assert old_log_probs.ndim == log_probs.ndim
        entropy = pi_dist.entropy()
        # Policy loss
        explained_variance = (
            1
            - (batch["ret"] - batch["val"]).pow(2).sum()
            / (batch["ret"] - batch["ret"].mean()).pow(2).sum()
        ).item()
        # entropy = self.actor.entropy(states).mean().item()
        kl = (old_log_probs - log_probs).mean()
        """
        info = dict(explained_variance=explained_variance,
                      entropy=entropy.detach().cpu().numpy(),
                      kl = (old_log_probs-log_probs).mean(),
                      pg_norm = pg_norm,
                      critic_loss = critic_loss.detach().cpu().numpy())
        """
        info = dict(
            explained_variance=0,
            entropy=0,
            kl=0,
            pg_norm=pg_norm.item(),
            critic_loss=critic_loss.item(),
        )
        return info

    def step(self, obs, deterministic=False):
        """
        TODO: Pay attention on state dimension [batch_size, state_size] or [state_size]
        """
        if len(obs.shape) == 3 or obs.ndim == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            value = self.critic(obs)
            pi, _ = self.actor(obs)
            action = pi.mode() if deterministic else pi.sample()
            log_prob = pi.log_probs(action)
            # action = pi.mean if deterministic else pi.sample()
            # log_prob = self.actor._log_prob_from_distribution(pi, action)
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

    def linesearch(self, npg, pg, states):
        parameters = self.actor.parameters()
        params_curr = parameters_to_vector(parameters)
        for k in range(self.num_backtrack):
            params_new = params_curr + self.alpha ** k * npg
            model_new = deepcopy(self.actor)
            model_param = model_new.parameters()
            vector_to_parameters(params_new, model_param)
            param_diff = params_new - params_curr
            surr_loss = torch.dot(pg, param_diff)
            kl_div = self.get_kl(model_new, states)
            if surr_loss >= 0 and kl_div <= self.delta:
                params_curr = params_new
                break
        return params_curr, (self.alpha ** k * npg).norm()


def conjugate_gradient(A, b, n):
    """
    Conjugate gradient algorithm
    - https://en.wikipedia.org/wiki/Conjugate_gradient_method
    - https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/trpo/trpo.py
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = torch.dot(r, r)
    for i in range(n):
        if callable(A):
            Ap = A(p)
        else:
            Ap = torch.matmul(A, p)
        alpha = rs / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_next = torch.dot(r, r)
        betta = rs_next / rs
        p = r + betta * p
        rs = rs_next
        if rs < 1e-10:
            break
    return x
