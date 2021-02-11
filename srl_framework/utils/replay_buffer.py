import numpy as np
import random
import scipy.signal
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from srl_framework.utils.cpc_crops import random_crop
import kornia
import torch.nn as nn


class Buffer():
    """
    On-policy agent buffer class using Generalized Advantage Estimation (GAE-Lambda)
    """

    def __init__(self, capacity, obs_shape, act_dim, device, **kwargs):
        self.capacity = capacity
        self.steps_per_epoch = kwargs['steps_per_epoch']
        self.device = device
        self.lam = kwargs['lam']
        self.gamma = kwargs['gamma']
        self.tau = kwargs['tau']
        self.contrastive = kwargs['contrastive']
        self.mini_batch_size = kwargs['mini_batch_size']
        self.image_size = kwargs['image_size']
        self.normalized_obs = kwargs['normalized_obs']
        self.drq = kwargs['data_regularization']
        image_pad = kwargs['image_pad']
        self.advantage_norm = False #kwargs['advantage_normalization']
        self.ptr, self.size, self.traj_start_ptr, self.epoch_ptr = 0, 0, 0, 0
         # the proprioceptive obs is stored as float32, pixels obs as uint8
        if len(obs_shape)>=3 and not self.normalized_obs:
            obs_dtype = np.uint8
        else:
            obs_dtype =np.float32

        if self.mini_batch_size:
            random_sampler = SubsetRandomSampler(range(self.steps_per_epoch))
            self.batch_sampler = BatchSampler(random_sampler, self.mini_batch_size, drop_last=True)    
        if self.drq: 
            self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))
         
        
        # Transition data
        self.obs = np.zeros(combined_shape(capacity, obs_shape), dtype=obs_dtype)
        self.obs_tp1 = np.zeros(combined_shape(capacity, obs_shape), dtype=obs_dtype)
        self.act = np.zeros(combined_shape(capacity,act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity), dtype=np.float32)
        self.done = np.zeros((capacity), dtype=np.float32)

        # Predictor data
        self.val = np.zeros((capacity), dtype=np.float32)
        self.log_prob = np.zeros((capacity), dtype=np.float32)

        # Rest
        self.return_mc = np.zeros((capacity), dtype=np.float32)
        self.return_gae = np.zeros((capacity), dtype=np.float32)
        self.advantage = np.zeros((capacity), dtype=np.float32)
    
    def __len__(self):
        return self.ptr

    def store(self, obs, act, rew, obs_tp1, done, val, log_prob):
        if self.ptr == self.epoch_ptr:
            if self.ptr + self.steps_per_epoch > self.capacity:
                self.ptr, self.traj_start_ptr, self.epoch_ptr = 0, 0, 0
        assert self.ptr < self.capacity     # buffer has to have room so you can store        
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.obs_tp1[self.ptr] = obs_tp1
        self.done[self.ptr] = done
        self.val[self.ptr] = val
        self.log_prob[self.ptr] = log_prob
        self.ptr += 1
        if self.size <= self.ptr: self.size += 1

    def finish_path(self, last_value=0, maximum_entropy = False):
        """
        Uses rewards and value estimates from the whole trajectory to compute advantage 
        estimates with GAE-Lambda, as well as compute the rewards-to-go for each state,
        to use as the targets for the value function.
        
        Parameters:
        ----------
            - 'last_value': Value function estimate V(s_t) for last state.
                            Should be 0 if the trajectory ended because the agent 
                            reached a terminal state (died), and otherwise V(s_t).
            - 'maximum_entropy': Set True if TODO         
        """
        
        # Get rewards and values of current episode
        idxs_episode = slice(self.traj_start_ptr, self.ptr)

        rewards = np.append(self.rew[idxs_episode], last_value)
        values = np.append(self.val[idxs_episode], last_value)

        # Maximum Entropy
        if maximum_entropy: rewards -= self.tau * self.log_prob[idxs_episode]
        
        # GAE-Lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage[idxs_episode] = discount_cumsum(deltas, self.gamma * self.lam)
            
        # Rewards-to-go, to be targets for the value function
        self.return_mc[idxs_episode] = discount_cumsum(rewards, self.gamma)[:-1]
        
        # TODO: CHECK!!
        self.return_gae[idxs_episode] = self.advantage[idxs_episode] + values[:-1]
        
        # Set starting index of next path to current index
        self.traj_start_ptr = self.ptr
    
    def sample(self, batch_size):
        #assert self.ptr-self.epoch_ptr == self.steps_per_epoch
        #self.ptr, self.traj_start_ptr = 0, 0
        # Set starting index of next path to current index

        #assert (self.ptr - self.epoch_ptr)%self.steps_per_epoch == 0
        assert (self.ptr - self.epoch_ptr) == self.steps_per_epoch  # check if reset was successfull
        start = self.epoch_ptr
        adv = torch.as_tensor(self.advantage[start:self.ptr], dtype=torch.float32, device=self.device)
        obs = self.obs[start:self.ptr]
        obs_tp1 = self.obs_tp1[start:self.ptr]
        if self.drq:
            obs_aug = obs.copy()
            obs_tp1_aug = obs_tp1.copy()
            obs= torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            obs_tp1=torch.as_tensor(obs_tp1, dtype=torch.float32, device=self.device)
            obs_aug= torch.as_tensor(obs_aug, dtype=torch.float32, device=self.device)
            obs_tp1_aug=torch.as_tensor(obs_tp1_aug, dtype=torch.float32, device=self.device)

            obs= self.aug_trans(obs)
            obs_tp1= self.aug_trans(obs_tp1)
            obs_aug= self.aug_trans(obs_aug)
            obs_tp1_aug= self.aug_trans(obs_tp1_aug)
        else:
            obs_aug= 0
            obs_tp1_aug= 0
            obs= torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            obs_tp1=torch.as_tensor(obs_tp1, dtype=torch.float32, device=self.device)    
        return dict(
            obs = obs,
            obs_tp1 = obs_tp1,
            obs_aug = obs_aug,
            obs_tp1_aug =obs_tp1_aug,
            act = torch.as_tensor(self.act[start:self.ptr], dtype=torch.float32, device=self.device),
            ret = torch.as_tensor(self.return_mc[start:self.ptr], dtype=torch.float32, device=self.device),
            ret_gae = torch.as_tensor(self.return_gae[start:self.ptr], dtype=torch.float32, device=self.device),
            adv = adv,
            val = torch.as_tensor(self.val[start:self.ptr], dtype=torch.float32, device=self.device),
            logp = torch.as_tensor(self.log_prob[start:self.ptr], dtype=torch.float32, device=self.device)
        )
    
    def reset_after_update(self):
        self.epoch_ptr = self.traj_start_ptr = self.ptr

    def sample_srl(self, batch_size, method = 'none', idxs=None):
        idxs = np.random.randint(0, self.size, size=batch_size) if idxs == None else idxs
        obs = self.obs[idxs]
        obs_tp1 = self.obs_tp1[idxs]
        obs = self.obs[idxs]
        obs_tp1 = self.obs_tp1[idxs]
        if self.drq:
            obs_aug = obs.copy()
            obs_tp1_aug = obs_tp1.copy()
            obs= torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            obs_tp1=torch.as_tensor(obs_tp1, dtype=torch.float32, device=self.device)
            obs_aug= torch.as_tensor(obs_aug, dtype=torch.float32, device=self.device)
            obs_tp1_aug=torch.as_tensor(obs_tp1_aug, dtype=torch.float32, device=self.device)

            obs= self.aug_trans(obs)
            obs_tp1= self.aug_trans(obs_tp1)
            obs_aug= self.aug_trans(obs_aug)
            obs_tp1_aug= self.aug_trans(obs_tp1_aug)
        else:
            obs_aug= 0
            obs_tp1_aug= 0
            obs= torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            obs_tp1=torch.as_tensor(obs_tp1, dtype=torch.float32, device=self.device) 
        
        return dict(
                obs = obs,
                obs_tp1 = obs_tp1,
                obs_aug = obs_aug,
                obs_tp1_aug =obs_tp1_aug,
                act=torch.as_tensor(self.act[idxs], dtype=torch.float32, device=self.device),
                rew=torch.as_tensor(self.rew[idxs], dtype=torch.float32, device=self.device),
                done=torch.as_tensor(self.done[idxs], dtype=torch.int, device=self.device)
                )
                
    def sample_cpc(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
      
        obs = self.obs[idxs]
        obs_tp1 = self.obs_tp1[idxs]
        pos = obs.copy()

        obs = random_crop(obs, self.image_size)
        obs_tp1 = random_crop(obs_tp1, self.image_size)
        pos = random_crop(pos, self.image_size)

        pos = torch.as_tensor(pos, device=self.device).float()
        #time_anchor=None, time_pos=None)
        
        return dict(obs= torch.as_tensor(obs, dtype=torch.float32, device=self.device),
                     obs_tp1=torch.as_tensor(obs_tp1, dtype=torch.float32, device=self.device),
                     act=torch.as_tensor(self.act[idxs], dtype=torch.float32, device=self.device),
                     rew=torch.as_tensor(self.rew[idxs], dtype=torch.float32, device=self.device),
                     done=torch.as_tensor(self.done[idxs], dtype=torch.int, device=self.device),
                     pos= torch.as_tensor(pos, dtype=torch.float32, device=self.device))

    def generator(self, rollouts, converted_batch = False):
        if self.mini_batch_size:
            for indices in self.batch_sampler:
                minibatch = dict()
                minibatch['obs'] = rollouts['obs'][indices]
                minibatch['obs_tp1'] = rollouts['obs_tp1'][indices]
                if self.drq: minibatch['obs_aug'] = rollouts['obs_aug'][indices]
                if self.drq: minibatch['obs_tp1_aug'] = rollouts['obs_tp1_aug'][indices]
                minibatch['act'] = rollouts['act'][indices]
                minibatch['ret'] = rollouts['ret'][indices]
                minibatch['ret_gae'] = rollouts['ret_gae'][indices]
                minibatch['adv'] = rollouts['adv'][indices]
                minibatch['val'] = rollouts['val'][indices]
                minibatch['logp'] = rollouts['logp'][indices]
                if converted_batch:
                    minibatch['critic_state_tp1'] = rollouts['critic_state_tp1'][indices]
                    minibatch['critic_state_t'] = rollouts['critic_state_t'][indices]
                    minibatch['actor_state_t'] = rollouts['actor_state_t'][indices]
                    minibatch['actor_state_tp1'] = rollouts['actor_state_tp1'][indices]

                yield minibatch
        else:
            yield rollouts


class ReplayBuffer():
    """
    A simple FIFO experience replay buffer.
    """

    def __init__(self, capacity, obs_shape, act_dim, device, **kwargs):
        self.capacity = capacity
        self.device = device
        self.ptr, self.size = 0, 0
        self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total
        self.normalized_obs = kwargs['normalized_obs']
        self.image_size = kwargs['image_size']
        self.contrastive = kwargs['contrastive']
        self.drq = kwargs['data_regularization']
        image_pad = kwargs['image_pad']
         # the proprioceptive obs is stored as float32, pixels obs as uint8

        if self.drq: 
            self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))
        if len(obs_shape)>=3 and not self.normalized_obs:
            obs_dtype = np.uint8
        else:
            obs_dtype =np.float32
        
        # Transition data
        self.obs = np.zeros(combined_shape(capacity, obs_shape), dtype=obs_dtype)
        self.obs_tp1 = np.zeros(combined_shape(capacity, obs_shape), dtype=obs_dtype)
        self.act = np.zeros(combined_shape(capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity,1), dtype=np.float32)
        self.done = np.zeros((capacity,1), dtype=np.float32)
    
    def __len__(self):
        return self.size

    def store(self, obs, act, rew, next_obs, done):           
        self.obs[self.ptr] = obs
        self.obs_tp1[self.ptr] = next_obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.done[self.ptr] = done
        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size+1, self.capacity)
        self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

    def sample(self, batch_size=32, idxs = None):
        idxs = np.random.randint(0, self.size, size=batch_size) if idxs == None else idxs
        idxs_hat = np.random.randint(idxs-32,idxs+32,1)
        idxs_hat = max(0,idxs_hat)
        obs = self.obs[idxs]
        obs_tp1 = self.obs_tp1[idxs]
        if self.drq:
            obs_aug = obs.copy()
            obs_tp1_aug = obs_tp1.copy()
            obs= torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            obs_tp1=torch.as_tensor(obs_tp1, dtype=torch.float32, device=self.device)
            obs_aug= torch.as_tensor(obs_aug, dtype=torch.float32, device=self.device)
            obs_tp1_aug=torch.as_tensor(obs_tp1_aug, dtype=torch.float32, device=self.device)

            obs= self.aug_trans(obs)
            obs_tp1= self.aug_trans(obs_tp1)
            obs_aug= self.aug_trans(obs_aug)
            obs_tp1_aug= self.aug_trans(obs_tp1_aug)

            return dict(obs= obs,
                     obs_tp1=obs_tp1,
                     obs_aug= obs_aug,
                     obs_tp1_aug=obs_tp1_aug,
                     act=torch.as_tensor(self.act[idxs], dtype=torch.float32, device=self.device),
                     rew=torch.as_tensor(self.rew[idxs], dtype=torch.float32, device=self.device),
                     done=torch.as_tensor(self.done[idxs], dtype=torch.int, device=self.device))
        else:
            return dict(obs= torch.as_tensor(obs, dtype=torch.float32, device=self.device),
                     obs_tp1=torch.as_tensor(obs_tp1, dtype=torch.float32, device=self.device),
                     act=torch.as_tensor(self.act[idxs], dtype=torch.float32, device=self.device),
                     rew=torch.as_tensor(self.rew[idxs], dtype=torch.float32, device=self.device),
                     done=torch.as_tensor(self.done[idxs], dtype=torch.int, device=self.device))
    
    def sample_srl(self, batch_size=32, method='none', idxs=None):
        if self.contrastive:
            return self.sample_cpc(batch_size)
        else:
            return self.sample(batch_size, idxs) 
    
    def sample_cpc(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
      
        obs = self.obs[idxs]
        obs_tp1 = self.obs_tp1[idxs]
        pos = obs.copy()

        obs = random_crop(obs, self.image_size)
        obs_tp1 = random_crop(obs_tp1, self.image_size)
        pos = random_crop(pos, self.image_size)

        pos = torch.as_tensor(pos, device=self.device).float()
        #time_anchor=None, time_pos=None)
        
        return dict(obs= torch.as_tensor(obs, dtype=torch.float32, device=self.device),
                     obs_tp1=torch.as_tensor(obs_tp1, dtype=torch.float32, device=self.device),
                     act=torch.as_tensor(self.act[idxs], dtype=torch.float32, device=self.device),
                     rew=torch.as_tensor(self.rew[idxs], dtype=torch.float32, device=self.device),
                     done=torch.as_tensor(self.done[idxs], dtype=torch.int, device=self.device),
                     pos= torch.as_tensor(pos, dtype=torch.float32, device=self.device))


class SequentialReplayBuffer:
    """
    Recurrent Replay Buffer
    """
    def __init__(self, capacity, obs_shape, act_dim, device, **kwargs):
        self.capacity = capacity // kwargs['ep_len']
        self.device = device
        self.ep_len = kwargs['ep_len']
        self.act_dim = act_dim
          # sampling index, whenever sampling is ordered
        self.sequential_len = kwargs['seq_len']
        self.safe_hidden = kwargs['safe_hidden']
        self.normalized_obs = kwargs['normalized_obs']
        self.image_size = kwargs['image_size']
        self.contrastive = kwargs['contrastive']
        self.drq = kwargs['data_regularization']

        if self.drq: 
            self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

        if len(obs_shape)>=3 and not self.normalized_obs:
            obs_dtype = np.uint8
        else:
            obs_dtype =np.float32

        self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total
        self.ptr, self.size, self.ep_ptr = 0, 0, 0
        self.full = False
        self.index, self.sub_index = 0, 0
        self.safe_hidden = False
        # Transition data       
        self.obs = np.zeros(combined_shape(self.capacity, combined_shape(self.ep_len, obs_shape)), dtype=obs_dtype)
        self.obs_tp1 = np.zeros(combined_shape(self.capacity, combined_shape(self.ep_len, obs_shape)), dtype=obs_dtype)
        self.seq_len = np.zeros(self.capacity, dtype=np.int)
        self.act = np.zeros(combined_shape(self.capacity, combined_shape(self.ep_len, act_dim)), dtype=np.float32)
        self.act_tm1 = np.zeros(combined_shape(self.capacity, combined_shape(self.ep_len+1, act_dim)), dtype=np.float32)
        self.rew = np.zeros(combined_shape(self.capacity, self.ep_len), dtype=np.float32)
        self.done = np.zeros(combined_shape(self.capacity, self.ep_len), dtype=np.float32)
        
    def store(self, obs, act, rew, obs_tp1, done, hidden_in=0, hidden_out=0):        
        self.obs[self.ptr][self.ep_ptr] = obs
        self.obs_tp1[self.ptr][self.ep_ptr] = obs_tp1
        self.act[self.ptr][self.ep_ptr] = act
        self.act_tm1[self.ptr][self.ep_ptr+1] = act # memory expensive but saves computation
        self.rew[self.ptr][self.ep_ptr] = rew
        self.done[self.ptr][self.ep_ptr] = done
        self.steps += 1
        self.ep_ptr += 1 
        self.seq_len[self.ptr] = self.ep_ptr
        if done:
            
            if self.ep_ptr < (self.sequential_len+5):
                print('override')
                self.ep_ptr = 0
                #override episode because it was too small
            else:
                self.ptr = (self.ptr+1) % self.capacity
                self.ep_ptr = 0
                if (self.ptr+1) % self.capacity == 0:
                    self.full = True
                self.episodes += 1
                self.size = self.capacity-1 if self.full else self.ptr-1

    def finish_path(self):
        print('finish_path')
        self.done[self.ptr][self.ep_ptr] = 0.0
        self.seq_len[self.ptr] = self.ep_ptr
        self.ep_ptr = 0
        
        if self.ep_ptr < (self.sequential_len+5):
            print('override')
            self.ep_ptr = 0
            #override episode because it was too small
        else:
            self.ptr = (self.ptr+1) % self.capacity
            self.ep_ptr = 0
            if (self.ptr+1) % self.capacity == 0:
                self.full = True
            self.episodes += 1
            self.size = self.capacity-1 if self.full else self.ptr-1

    def update_index(self, batch_size=32, random=True):
        self.index = min(self.index + batch_size, self.size) % self.size \
            if self.sub_index+self.sequential_len >= self.seq_len[self.index] else self.index
        self.sub_index = 0 if random else min(self.sub_index + self.sequential_len, self.seq_len[self.index]) % self.seq_len[self.index]
        self.index = 0 if random else self.index

    def sample_srl(self, batch_size=32, method = 'none'):
        return self.sample(batch_size, random=True, sequential=True, method = method)

    def sample(self, batch_size=32, random=True, sequential=True, method='none'):
        
        if random:
            idxs = np.array([i for i,v in enumerate(self.seq_len >= self.sequential_len+5) if v])
            idxs = np.random.choice(idxs, batch_size)
            self.index, self.sub_index = (0, 0)
            
            sub_idx = [np.random.randint(0, min(self.ep_len-self.sequential_len, self.seq_len[i]-self.sequential_len)) for i in idxs]
        else:
            idxs = np.arange(self.index, min(self.index+batch_size, size_temp))
            sub_idx = [self.sub_index]

        if sequential: 
            sub_idx_range = [range(sub_i, min(sub_i+self.sequential_len, self.seq_len[idxs[i]])) for i,sub_i in enumerate(sub_idx)]

        self.update_index(batch_size=batch_size, random=random)

        if method =='slac':
            # observation input is obs_t-(k+1):t and a_t-k:t --> actually they termed the timestamp a little differently ... TODO 
            obs_tm8_tm1 = np.array([self.obs[idxs[i]][sub_idx_range[i]] for i in range(batch_size)])
            obs_t = np.expand_dims(np.array([self.obs_tp1[idxs[i]][sub_idx_range[i][-1]] for i in range(batch_size)]),axis=1)
            obs = np.concatenate((obs_tm8_tm1,obs_t), axis = 1)
            act=np.array([self.act[idxs[i]][sub_idx_range[i]] for i in range(batch_size)])
            test = 1
        else: 
            obs=np.array([self.obs[idxs[i]][sub_idx_range[i]] for i in range(batch_size)])
        
        if self.safe_hidden:
            batch = dict(hidden_in=(np.array([self.hidden_in[0][idxs[i]][sub_idx[i]] for i in range(batch_size)]),
                                np.array([self.hidden_in[1][idxs[i]][sub_idx[i]] for i in range(batch_size)])),
                     hidden_out=(np.array([self.hidden_out[0][idxs[i]][sub_idx[i]] for i in range(batch_size)]),
                                 np.array([self.hidden_out[1][idxs[i]][sub_idx[i]] for i in range(batch_size)])),
                     obs=obs,
                     obs_tp1=np.array([self.obs_tp1[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                     act=np.array([self.act[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                     act_tm1=np.array([self.act_tm1[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                     rew=np.array([self.rew[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                     done=np.array([self.done[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]))
        else:
            batch = dict(obs=obs,
                     obs_tp1=np.array([self.obs_tp1[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                     act=np.array([self.act[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                     rew=np.array([self.rew[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                     done=np.array([self.done[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]))
        return {k: torch.as_tensor(v, dtype=torch.float32, device=torch.device(self.device)) for k, v in batch.items()}

    def sample_slac(self, batch_size=32, random=True, sequential=True, slac=True):
        if random:
            size_temp = self.size-2 if self.seq_len[self.size-1]<(self.sequential_len+5) else self.size-1
            idxs = np.random.randint(0, size_temp, size=batch_size)
            self.index, self.sub_index = (0, 0)
            sub_idx = [np.random.randint(0, min(self.ep_len-self.sequential_len, self.seq_len[i]-self.sequential_len)) for i in idxs]
        else:
            idxs = np.arange(self.index, min(self.index+batch_size, size_temp))
            sub_idx = [self.sub_index]

        if sequential: 
            sub_idx_range = [range(sub_i, min(sub_i+self.sequential_len, self.seq_len[idxs[i]])) for i,sub_i in enumerate(sub_idx)]
        
        self.update_index(batch_size=batch_size, random=random)
        obs_tm8_tm1 = np.array([self.obs[idxs[i]][sub_idx_range[i]] for i in range(batch_size)])
        obs_t = np.expand_dims(np.array([self.obs_tp1[idxs[i]][sub_idx_range[i][-1]] for i in range(batch_size)]),axis=1)
        obs = np.concatenate((obs_tm8_tm1,obs_t), axis = 1)
        batch = dict(obs=obs,
                     obs_tp1= np.array([self.obs_tp1[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                     act=np.array([self.act[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                     rew=np.array([self.rew[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]),
                     done=np.array([self.done[idxs[i]][sub_idx_range[i]] for i in range(batch_size)]))
        return {k: torch.as_tensor(v, dtype=torch.float32, device=torch.device(self.device)) for k, v in batch.items()}

    def dump(self, save_dir):
        fn = os.path.join(save_dir, "replay_buffer.pkl")
        with open(fn, 'wb+') as f:
            pickle.dump(self, f)
        print(f"Buffer dumped to {fn}")