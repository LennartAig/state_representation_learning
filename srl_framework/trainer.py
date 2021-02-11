import os
import inspect
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from tqdm import tqdm
import copy as copy

# Buffer
from srl_framework.envs.environment import make_env
from srl_framework.utils.replay_buffer import (
    Buffer,
    ReplayBuffer,
    SequentialReplayBuffer,
)
from srl_framework.utils.logger import create_epoch_logger
from srl_framework.utils.config import get_cfg_defaults
from srl_framework.rl.get_rl_agent import get_rl_agent_by_name
from srl_framework.utils.utilities import set_seeds
from srl_framework.srl.srl_module import SRLModule


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Trainer:
    """
    Base Learner Class: Combines SRL and RL objectives based on given parameters
    """

    def __init__(self, args):
        self.param, parameter_path = self.load_parameters(args.experiment_file)
        self.param.freeze()

        self.num_epochs = args.epochs
        self.steps_per_epoch = args.steps_per_epoch
        # srl_rl initial steps: initial updates srl module and rl agent simultaniously
        self.srl_rl_initial_update_steps = args.initial_update_steps
        # srl initial steps: initial updates srl module only
        self.srl_initial_update_steps = self.param.SRL.INITIAL_UPDATE_STEPS
        if self.srl_rl_initial_update_steps > 0:
            self.srl_initial_update_steps = 0
        self.buffer_size = args.buffer_size
        self.sim_env = args.env_type
        self.video_freq = args.video_freq
        self.test_episodes = args.test_episodes
        self.initial_srl_steps = self.param.SRL.UPDATE_AFTER
        self.initial_rl_steps = self.param.RL.UPDATE_AFTER
        self.rl_delayed_start = self.param.RL.DELAYED_START
        # Seed setting
        set_seeds(args.seed)

        # Environment
        curl = (
            "LATENT" in self.param.SRL.MODELS and self.param.SRL.LATENT.TYPE == "CURL"
        )
        image_size = args.render_image_size if curl else args.input_image_size
        self.env, self.envtype, self.obs_shape, self.act_dim, self.act_limit = make_env(
            args, image_size
        )
        try:
            self.max_eplen = self.env._max_episode_steps
        except:
            self.max_eplen = 500

        # Test action and observation shape
        act = self.env.action_space.sample()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Init Logger
        self.logger = create_epoch_logger(self.param, parameter_path, args)

        obs = self.env.reset()
        if obs.ndim >= 3:
            obs = torch.FloatTensor(obs).to(self.device)
            self.logger.log_image("observation", obs, 0)

        self.encoder_args = {
            "architecture": self.param.ENCODER.ARCHITECTURE,
            "squash_latent": self.param.ENCODER.SQUASHED_LATENT,
            "normalize": self.param.ENCODER.NORMALIZED_LATENT,
            "conv_init": self.param.ENCODER.CONV_INIT,
            "linear_init": self.param.ENCODER.LINEAR_INIT,
            "activation": self.param.ENCODER.ACTIVATION,
            "batchnorm": self.param.ENCODER.BATCHNORM,
            "pool": False,
            "dropout": self.param.ENCODER.DROPOUT,
            "bias": self.param.ENCODER.BIAS,
            "hidden_dim": self.param.ENCODER.HIDDEN_DIM,
        }

        critic_args = {
            "architecture": self.param.RL.CRITIC.ARCHITECTURE,
            "activation": self.param.RL.CRITIC.ACTIVATION,
            "batchnorm": self.param.RL.CRITIC.BATCHNORM,
            "dropout": self.param.RL.CRITIC.DROPOUT,
            "init": self.param.RL.CRITIC.INIT,
        }

        actor_args = {
            "architecture": self.param.RL.CRITIC.ARCHITECTURE,
            "activation": self.param.RL.CRITIC.ACTIVATION,
            "batchnorm": self.param.RL.CRITIC.BATCHNORM,
            "dropout": self.param.RL.CRITIC.DROPOUT,
            "init": self.param.RL.CRITIC.INIT,
        }

        # Init State Representation Learning
        print("--------------------SRL MODELS---------------------")
        kwargs = {
            "encoder_args": self.encoder_args,
            "device": self.device,
            "obs_shape": (self.obs_shape[0], image_size, image_size)
            if curl
            else self.obs_shape,
            "act_dim": self.act_dim,
            "envtype": self.envtype,
            "normalized_obs": args.normalize_obs,
        }
        self.srl_module = SRLModule(self.param.SRL, **kwargs)
        self.srl_module.apply(weight_init)
        self.srl_info = self.srl_module.get_info()
        self.srl_batchsize = self.param.SRL.BATCH_SIZE
        self.srl_only_pretrainig = self.param.SRL.ONLY_PRETRAINING
        self.joint_training = self.param.SRL.JOINT_TRAINING
        print(self.srl_module)

        # Init Reinforcement Learning Agent
        print(
            "--------------------RL AGENT: {} ---------------------".format(
                self.param.RL.NAME
            )
        )
        rl_algo, self.agent_class = get_rl_agent_by_name(self.param.RL.NAME)
        base_rl_kwargs = {
            **{
                "feature_dim": self.param.RL.FEATURE_DIM,
                "encoder_args": self.encoder_args,
                "device": self.device,
                "img_channels": self.obs_shape[0],
                "action_limit": self.act_limit,
                "action_dim": self.act_dim,
                "envtype": self.envtype,
                "normalized_obs": args.normalize_obs,
                "img_size": self.obs_shape[-1],
                "data_regularization": self.srl_info["data_augmentation"],
                "actor_args": actor_args,
                "critic_args": critic_args,
                "schedule_param": args.epochs,
            },
            **self.srl_info,
        }
        if self.param.RL.NAME == "sac":
            rl_kwargs = {
                "polyak": self.param.RL.SAC.POLYAK,
                "gamma": self.param.RL.SAC.GAMMA,
                "target_update_freq": self.param.RL.SAC.TARGET_UPDATE_FREQ,
                "target_encoder": self.srl_module.latent_model.encoder_targ
                if self.srl_info["contrastive"]
                else None,
            }
            self.batchsize = self.param.RL.BATCH_SIZE
            if self.srl_info["contrastive"]:
                self.srl_module.do_target_update = True
        elif self.param.RL.NAME == "trpo":
            rl_kwargs = {
                "entropy_coef": self.param.RL.TRPO.ENTROPY_COEF,
                "cg_damping": self.param.RL.TRPO.CG_DAMPING,
                "cg_steps": self.param.RL.TRPO.CG_STEPS,
                "value_epochs": self.param.RL.TRPO.VALUE_EPOCHS,
                "num_backtrack": self.param.RL.TRPO.NUM_BACKTRACK,
                "delta": self.param.RL.TRPO.DELTA,
                "alpha": self.param.RL.TRPO.ALPHA,
                "fixed_std": self.param.RL.TRPO.FIXED_STD,
                "squashed": self.param.RL.TRPO.SQUASHED,
                "lr_scheduler": self.param.RL.TRPO.LR_SCHEDULER,
            }
            self.on_policy_update_steps = 1
            self.batchsize = self.steps_per_epoch
        elif self.param.RL.NAME == "ppo":
            rl_kwargs = {
                "fixed_std": self.param.RL.PPO.FIXED_STD,
                "clip_ratio": self.param.RL.PPO.CLIPPED_RATIO,
                "policy_epochs": self.param.RL.PPO.POLICY_EPOCHS,
                "critic_epochs": self.param.RL.PPO.VALUE_EPOCHS,
                "early_stopping": self.param.RL.PPO.EARLY_STOPPING,
                "max_kl_div": self.param.RL.PPO.MAX_KL_DIV,
                "squashed": self.param.RL.PPO.SQUASHED,
                "advantage_normalization": self.param.RL.ADVANTAGE_NORM,
                "lr_scheduler": self.param.RL.PPO.LR_SCHEDULER,
            }
            self.on_policy_update_steps = 1
            self.batchsize = self.steps_per_epoch
            self.target_kl = None
        elif self.param.RL.NAME == "ppo2":
            rl_kwargs = {
                "fixed_std": self.param.RL.PPO.FIXED_STD,
                "clip_ratio": self.param.RL.PPO.CLIPPED_RATIO,
                "use_value_clipping": self.param.RL.PPO.CLIPPED_VALUE,
                "clipping_range": self.param.RL.PPO.CLIP_VALUE_RANGE,
                "squashed": self.param.RL.PPO.SQUASHED,
                "use_grad_clipping": self.param.RL.PPO.GRAD_CLIPPING,
                "max_grad_norm": self.param.RL.PPO.MAX_GRAD_NORM,
                "value_coef": self.param.RL.PPO.VALUE_COEFFICIENT,
                "entropy_coef": self.param.RL.PPO.ENTROPY_COEFFICIENT,
                "cutoff_coef": self.param.RL.PPO.CUTOFF_COEFFICIENT,
                "advantage_normalization": self.param.RL.ADVANTAGE_NORM,
                "early_stopping": self.param.RL.PPO.EARLY_STOPPING,
                "max_kl_div": self.param.RL.PPO.MAX_KL_DIV,
                "lr_scheduler": self.param.RL.PPO.LR_SCHEDULER,
            }
            self.on_policy_update_steps = self.param.RL.PPO.TRAINING_EPOCHS
            self.batchsize = self.steps_per_epoch
            self.target_kl = (
                self.param.RL.PPO.MAX_KL_DIV
                if self.param.RL.PPO.EARLY_STOPPING
                else None
            )
        else:
            raise NotImplementedError
        kwargs = {**base_rl_kwargs, **rl_kwargs}
        encoder = (
            self.srl_module.latent_model.encoder if self.srl_info["latent"] else None
        )
        self.rl_agent = rl_algo(param=self.param.RL, encoder=encoder, **kwargs)
        self.srl_module.train()
        print("-------------------------------------------------")

        # Init Replay Buffer. Depends wether RL Agent is on, or off policy:
        save_act_dim = 1 if self.envtype == "Discrete" else self.act_dim
        buffer_kwargs = dict(
            steps_per_epoch=self.steps_per_epoch,
            lam=0.95,
            gamma=0.99,
            tau=1,
            mini_batch_size=self.param.RL.PPO.MINI_BATCH_SIZE,  # if from_pixels else None,
            normalized_obs=args.normalize_obs,
            hidden_dim=self.param.SRL.RECURRENT.HIDDEN_SIZE,
            ep_len=self.max_eplen,
            recurrent_net_depth=self.param.SRL.SSM.LSTM_LAYERS,
            seq_len=self.srl_info["seq_len"],
            safe_hidden=False,
            image_size=image_size if self.srl_info["contrastive"] else None,
            contrastive=self.srl_info["contrastive"],
            data_regularization=self.srl_info["data_augmentation"],
            advantage_normalization=self.param.RL.ADVANTAGE_NORM,
            image_pad=4,
        )

        # self.rl_agent_sac = SacAeAgent(obs_shape=self.obs_shape,action_shape=self.act_dim, device=self.device, encoder=self.srl_module.latent_model.encoder)
        if self.agent_class == "on_policy":
            self.memory = Buffer(
                capacity=self.buffer_size,
                obs_shape=self.env.observation_space.shape,
                act_dim=self.act_dim,
                device=self.device,
                **buffer_kwargs,
            )
        elif self.agent_class == "off_policy" and not self.srl_info["sequential"]:
            self.memory = ReplayBuffer(
                capacity=self.buffer_size,
                obs_shape=self.env.observation_space.shape,
                act_dim=self.act_dim,
                device=self.device,
                **buffer_kwargs,
            )
        elif self.agent_class == "off_policy" and self.srl_info["sequential"]:
            self.memory = SequentialReplayBuffer(
                capacity=self.buffer_size,
                obs_shape=self.env.observation_space.shape,
                act_dim=self.act_dim,
                device=self.device,
                **buffer_kwargs,
            )
        else:
            raise NotImplementedError

    def step(self, obs, steps=-1, deterministic=False):

        if steps > self.rl_delayed_start or steps == -1:  # Action sampled by Policy
            state, features = self.srl_module.get_state(obs)
            act, log_prob, val = self.rl_agent.step(state, deterministic)
            if self.srl_info["state_type"] == "feature_hidden":
                self.srl_module.state_space_model.infer(
                    features, torch.FloatTensor(act).to(self.device)
                )
            if self.envtype == "Discrete":
                act = act[0]

        else:  # Random action sampled by environment
            act = self.env.action_space.sample()
            # only needed in some case. Restructure !!
            if self.agent_class == "on_policy":
                with torch.no_grad():
                    state, features = self.srl_module.get_state(obs)
                    _, log_prob = self.rl_agent.actor(
                        state, torch.FloatTensor(act).to(self.device)
                    )
                    log_prob = log_prob.cpu().numpy()
                    val = self.rl_agent.critic(state).cpu().numpy()
            else:
                log_prob = 0
                val = 0

        # Clip actions for ppo and trpo TODO
        if (
            self.agent_class == "on_policy"
            and self.sim_env == "dmc"
            and self.envtype == "Box"
        ):
            act = np.clip(act, self.env.action_space.low, self.env.action_space.high)

        return act, log_prob, val

    def run(self):

        # Setup SRL loss logging
        self.rl_info = self.rl_agent.get_loss_info_dict()
        self.srl_loss_info = self.srl_module.get_loss_info_dict()
        self.srl_info_img = None

        # Start RL process
        for epoch in range(self.num_epochs):
            obs, ep_ret, ep_len = self.env.reset(), 0, 0
            if self.srl_info["state_type"] == "feature_action_seq":
                obs_deque, action_deque = self.reset_deque(obs, True)
            for t in range(self.steps_per_epoch):

                total_steps = 1 + t + epoch * self.steps_per_epoch

                if self.srl_info["state_type"] == "feature_action_seq":
                    obs_in = (obs_deque, action_deque)
                else:
                    obs_in = obs

                act, log_prob, val = self.step(obs_in, total_steps, deterministic=False)

                obs_tp1, rew, done, _ = self.env.step(act)
                done_bool = 0.0 if ep_len < 25 and done else float(done)
                ep_ret += rew
                ep_len += 1

                if self.agent_class == "off_policy":
                    self.optimize_off_policy(total_steps, epoch)
                if self.agent_class == "on_policy":
                    self.memory.store(obs, act, rew, obs_tp1, done_bool, val, log_prob)
                else:
                    self.memory.store(obs, act, rew, obs_tp1, done_bool)

                obs = obs_tp1

                if self.srl_info["state_type"] == "feature_action_seq":
                    obs_deque.append(obs)
                    action_deque.append(act)

                if done or (ep_len == (self.max_eplen)):
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    if self.agent_class == "on_policy":
                        if not done:
                            obs = torch.FloatTensor(obs).to(self.device)
                            if len(obs.shape) == 3:
                                obs = obs.unsqueeze(0)
                            # feature = self.srl_module.latent_model.get_state(obs, grad=False) if self.srl_info['latent'] else obs
                            _, _, value = self.rl_agent.step(obs)
                        else:
                            value = 0
                        self.memory.finish_path(value)
                    if self.srl_info["state_type"] == "feature_action_seq":
                        obs_deque, action_deque = self.reset_deque(obs, True)
                        if not done:
                            self.memory.finish_path()
                    obs, ep_ret, ep_len = self.env.reset(), 0, 0

            if (
                self.agent_class == "off_policy"
                and self.srl_info["sequential"]
                and not done
            ):
                self.memory.finish_path()
            self.logger.store(EpRet=ep_ret, EpLen=ep_len)
            if self.agent_class == "on_policy":
                self.optimize_on_policy(total_steps, epoch)

            # Just hack to use logger
            if total_steps <= self.initial_srl_steps:
                self.logger.store(**self.srl_loss_info)
            if total_steps <= self.initial_rl_steps:
                self.logger.store(**self.rl_info)

            # Logging
            self.logger.log_tabular("Epoch", epoch, tensorboard=False)
            self.logger.log_tabular("EpRet", epoch=epoch, with_min_and_max=True)

            for key, _ in self.srl_loss_info.items():
                self.logger.log_tabular(
                    key, epoch=epoch, with_min_and_max=False, tensorboard=True
                )
            for key, _ in self.rl_info.items():
                self.logger.log_tabular(
                    key, epoch=epoch, with_min_and_max=False, tensorboard=True
                )

            self.test_agent(epoch)
            self.logger.log_tabular(
                "TestEpRet", epoch=epoch, with_min_and_max=True, tensorboard=True
            )
            self.logger.log_tabular(
                "TestEpLen", epoch=epoch, average_only=True, tensorboard=True
            )
            self.logger.dump_tabular()

        self.env.close()

    def test_agent(self, epoch=0):
        log_video = True if epoch > 10 and self.video_freq > 0 else False
        test_env = self.env
        if log_video and epoch % self.video_freq == 0:
            self.logger.video_rec.init(enabled=True)
        for test_ep in range(self.test_episodes):
            test_env = self.env
            test_env.seed(test_ep)
            obs, done, ep_ret, ep_len = test_env.reset(), False, 0, 0
            if self.srl_info["state_type"] == "feature_action_seq":
                obs_deque, action_deque = self.reset_deque(obs, True)

            while not (done or (ep_len == self.max_eplen)):

                if self.srl_info["state_type"] == "feature_action_seq":
                    obs_in = (obs_deque, action_deque)
                else:
                    obs_in = obs

                act, _, _ = self.step(obs_in, steps=-1, deterministic=True)
                obs, reward, done, _ = test_env.step(act)
                if log_video and epoch % self.video_freq == 0:
                    self.logger.video_rec.record(test_env, camera_id=8)
                ep_ret += reward
                ep_len += 1
                if self.srl_info["state_type"] == "feature_action_seq":
                    obs_deque.append(obs)
                    action_deque.append(act)
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            if log_video and epoch % self.video_freq == 0:
                self.logger.video_rec.save("%d.mp4" % epoch)

    def optimize_off_policy(self, steps, epoch=0):
        # if steps > self.initial_rl_stepsor steps > self.initial_srl_steps:
        #    batch = self.memory.sample_srl(self.param.RL.BATCH_SIZE) TODO!!

        if (
            steps >= self.initial_rl_steps
            and steps >= self.initial_srl_steps
            and self.srl_rl_initial_update_steps > 0
        ):
            bar = tqdm(range(self.srl_rl_initial_update_steps))
            for step in bar:
                bar.set_description("INITIAL TRAINING OF SRL AND RL")
                batch = self.memory.sample_srl(
                    self.batchsize, method=self.srl_info["method"]
                )
                batch_rl = self.srl_module.convert_batch(batch.copy())
                if (
                    self.srl_info["use_srl"]
                    and self.srl_info["rl_loss"]
                    and self.srl_info["state_type"] == "state"
                ):
                    self.rl_info = self.rl_agent.optimize(
                        batch_rl,
                        steps=step,
                        latent_optimizer=self.srl_module.latent_model.optimizer,
                    )
                else:
                    self.rl_info = self.rl_agent.optimize(batch_rl, steps=step)
                if self.srl_info["use_srl"]:
                    self.srl_loss_info, srl_info_img = self.srl_module.optimize(
                        batch, step=step
                    )
                    self.logger.store(**self.srl_loss_info)
                self.logger.store(**self.rl_info)
            self.srl_rl_initial_update_steps = 0

        if steps >= self.initial_rl_steps or steps >= self.initial_srl_steps:
            batch = self.memory.sample_srl(
                self.batchsize, method=self.srl_info["method"]
            )

        if steps > self.initial_rl_steps and steps % self.param.RL.UPDATE_EVERY == 0:
            batch_rl = self.srl_module.convert_batch(batch.copy())
            if (
                self.srl_info["use_srl"]
                and self.srl_info["rl_loss"]
                and self.srl_info["state_type"] == "state"
            ):
                self.rl_info = self.rl_agent.optimize(
                    batch_rl,
                    steps=steps,
                    latent_optimizer=self.srl_module.latent_model.optimizer,
                )
            else:
                self.rl_info = self.rl_agent.optimize(batch_rl, steps=steps)
            self.logger.store(**self.rl_info)

        if steps >= self.initial_srl_steps and self.srl_info["use_srl"]:
            if self.srl_initial_update_steps > 0:
                bar = tqdm(range(self.srl_initial_update_steps))
                for step in bar:
                    bar.set_description("INITIAL TRAINING OF SRL")
                    batch = self.memory.sample_srl(
                        self.batchsize, method=self.srl_info["method"]
                    )
                    self.srl_loss_info, srl_info_img = self.srl_module.optimize(
                        batch, step=step
                    )
                    self.logger.store(**self.srl_loss_info)
                if srl_info_img["reconstructed"]:
                    self.logger.log_image(
                        "observation and reconstruction",
                        torch.cat(
                            (srl_info_img["obs_t"], srl_info_img["decoded_t"]), 0
                        ),
                        epoch,
                    )
                self.srl_initial_update_steps = 0
            else:
                if steps % self.param.SRL.UPDATE_EVERY == 0:
                    # batch = self.memory.sample_srl(self.param.RL.BATCH_SIZE, method = self.srl_info['method'])
                    self.srl_loss_info, srl_info_img = self.srl_module.optimize(batch)
                    self.logger.store(**self.srl_loss_info)

        if (
            steps % self.steps_per_epoch == 0
            and steps > self.initial_srl_steps
            and self.srl_info["use_srl"]
        ):
            if srl_info_img["reconstructed"]:
                self.logger.log_image(
                    "observation and reconstruction",
                    torch.cat((srl_info_img["obs_t"], srl_info_img["decoded_t"]), 0),
                    epoch,
                )

    def optimize_on_policy(self, steps, epoch=0):
        # if steps >= self.initial_rl_steps or steps >= self.initial_srl_steps:
        #    batch = self.memory.sample(self.batchsize)
        if (
            steps >= self.initial_srl_steps
            and self.srl_info["use_srl"]
            and self.srl_initial_update_steps > 0
        ):
            bar = tqdm(range(self.srl_initial_update_steps))
            for _ in bar:
                bar.set_description("INITIAL TRAINING OF SRL")
                batch = self.memory.sample_srl(self.srl_batchsize)
                self.srl_loss_info, srl_info_img = self.srl_module.optimize(batch)
                self.logger.store(**self.srl_loss_info)
            self.srl_initial_update_steps = 0

        elif steps >= self.initial_srl_steps and self.srl_info["use_srl"]:
            if steps >= self.initial_rl_steps and self.joint_training:
                pass
            elif not self.srl_only_pretrainig:
                for _ in range(0, int(self.steps_per_epoch / self.srl_batchsize)):
                    batch = self.memory.sample_srl(self.srl_batchsize)
                    self.srl_loss_info, srl_info_img = self.srl_module.optimize(batch)
                    self.logger.store(**self.srl_loss_info)
            else:
                self.logger.store(**self.srl_loss_info)
        if steps >= self.initial_rl_steps:
            batch = self.memory.sample(self.batchsize)

            for update_epoch in range(self.on_policy_update_steps):
                approx_kl_divs = []
                generator = self.memory.generator(batch)
                i = 0
                for minibatch in generator:
                    if self.srl_info["use_srl"] and self.joint_training:
                        self.srl_loss_info, srl_info_img = self.srl_module.optimize(
                            minibatch
                        )
                        self.logger.store(**self.srl_loss_info)
                    minibatch = self.srl_module.convert_batch(minibatch)
                    self.rl_info = self.rl_agent.optimize(minibatch, num_minibatch=i)
                    self.logger.store(**self.rl_info)
                    i += 1
                    approx_kl_divs.append(self.rl_info["KL"])
                if (
                    self.target_kl is not None
                    and np.mean(approx_kl_divs) > 1.5 * self.target_kl
                ):
                    print(
                        f"Early stopping at step {update_epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}"
                    )
                    break

        self.memory.reset_after_update()

        if "srl_info_img" in locals():
            if srl_info_img["reconstructed"]:
                self.logger.log_image(
                    "observation and reconstruction",
                    torch.cat((srl_info_img["obs_t"], srl_info_img["decoded_t"]), 0),
                    epoch,
                )

    def reset_deque(self, obs, slac=False):
        """
        reset state and action deque
        """
        if slac:
            obs_deque = deque(maxlen=self.srl_info["seq_len"])
            action_deque = deque(maxlen=self.srl_info["seq_len"] - 1)
            for _ in range(self.srl_info["seq_len"] - 1):
                obs_deque.append(np.zeros(self.obs_shape, dtype=np.float32))
                action_deque.append(np.zeros(self.act_dim, dtype=np.float32))
            obs_deque.append(obs)
        else:
            obs_deque = deque(maxlen=self.srl_info["seq_len"])
            action_deque = deque(maxlen=self.srl_info["seq_len"])
        return obs_deque, action_deque

    def deque_to_seq(self, obs):
        """
        Input: dque
        Returns: Torchtensors on device
        """
        obs_deque, act_deque = obs[0], obs[1]
        obs = np.array(obs_deque, dtype=np.float32)
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        action = np.array(act_deque, dtype=np.float32)
        action = torch.FloatTensor(action).to(self.device).unsqueeze(0)
        return obs, action

    def load_parameters(self, experiment_name):
        """
        Loads parameters from experiment yaml file
        """
        experiment_parameter_path = (
            os.path.dirname(inspect.getfile(self.__class__))
            + "/experiments/"
            + experiment_name
        )
        with open(experiment_parameter_path) as f:
            parameters = get_cfg_defaults()
            parameters.merge_from_file(experiment_parameter_path)
            return parameters, experiment_parameter_path
