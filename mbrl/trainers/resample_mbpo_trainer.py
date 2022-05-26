from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn

import mbrl.torch_modules.utils as ptu
from mbrl.utils.eval_util import create_stats_ordered_dict
from mbrl.trainers.sac_trainer import SACTrainer
from mbrl.global_config import *
import pdb
import copy

class ResampleMBPOTrainer(SACTrainer):
    def __init__(self, 
            env, 
            model,
            policy,
            qf,
            resample_mode='from_ensemble',
            **sac_trainer_kwargs
        ):
        super().__init__(env, policy, qf, **sac_trainer_kwargs)
        self.model = model
        self.resample_mode = resample_mode
        self.reward_f, self.done_f = env.get_reward_done_function()

    def train(self, real_data, imagined_data):
        return self.train_from_numpy_batch(real_data, imagined_data)

    def train_from_numpy_batch(self, np_real_batch, np_imagined_data):
        np_real_batch = ptu.np_to_pytorch_batch(np_real_batch)
        np_imagined_data = ptu.np_to_pytorch_batch(np_imagined_data)
        return self.train_from_torch_batch(np_real_batch, np_imagined_data)

    def compute_q_target(self, next_obs, rewards, terminals):
        alpha = self._get_alpha()

        next_action, next_policy_info = self.policy.action(
            next_obs, reparameterize=False, return_log_prob=True,
        )
        log_prob_next_action = next_policy_info['log_prob']
        target_q_next_action = self.qf.value(next_obs, 
                                        next_action, 
                                        use_target_value=True, 
                                        return_info=False) - alpha * log_prob_next_action

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_next_action
        return q_target

        
    def train_from_torch_batch(self, batch, imagined_batch):
        real_obs = batch['observations']
        n_real = real_obs.shape[0]

        imagined_obs = imagined_batch['observations']
        obs = torch.cat([real_obs, imagined_obs], 0)

        real_actions = batch['actions']
        imagined_actions = imagined_batch['actions']
        actions = torch.cat([real_actions, imagined_actions], 0)

        real_next_obs = batch['next_observations']
        real_rewards = batch['rewards']
        real_terminals = batch['terminals']

        compute_reward = True
        compute_terminal = True

        if self.resample_mode == 'from_single':
            imagined_next_obs_mean = imagined_batch['next_obs_mean']
            imagined_next_obs_std = imagined_batch['next_obs_std']
            if 'reward_mean' in imagined_batch:
                compute_reward = False
                imagined_reward_mean = imagined_batch['reward_mean']
                imagined_reward_std = imagined_batch['reward_std']
        elif self.resample_mode == 'from_ensemble':
            imagined_next_obs_mean = imagined_batch['ensemble_next_obs_mean']
            imagined_next_obs_std = imagined_batch['ensemble_next_obs_std']
            assert imagined_next_obs_mean.shape == imagined_next_obs_std.shape and \
                imagined_next_obs_mean.dim() == 3
            imagined_batch_size = imagined_next_obs_mean.shape[0]
            
            indices = np.random.choice(self.model.elite_indices, imagined_batch_size)
            n_arange = np.arange(imagined_batch_size)

            imagined_next_obs_mean = imagined_next_obs_mean[n_arange, indices]
            imagined_next_obs_std = imagined_next_obs_std[n_arange, indices]
            
            if 'ensemble_reward_mean' in imagined_batch:
                compute_reward = False
                imagined_reward_mean = imagined_batch['ensemble_reward_mean'][n_arange, indices]
                imagined_reward_std = imagined_batch['ensemble_reward_std'][n_arange, indices]

        imagined_next_obs = ptu.normal(
            imagined_next_obs_mean, 
            imagined_next_obs_std
        )
        next_obs = torch.cat([real_next_obs, imagined_next_obs], 0)

        if compute_reward:
            imagined_rewards = self.reward_f(imagined_obs, imagined_actions, imagined_next_obs)
        else:
            imagined_rewards = ptu.normal(
            imagined_reward_mean, 
            imagined_reward_std
        )
        rewards = torch.cat([real_rewards, imagined_rewards], 0)

        if compute_terminal:
            imagined_terminals = self.done_f(imagined_obs, imagined_actions, imagined_next_obs)
        terminals = torch.cat([real_terminals, imagined_terminals], 0)

        q_target = self.compute_q_target(next_obs, rewards, terminals)
        qf_loss, train_qf_info = self.compute_qf_loss(obs, actions, q_target)

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        if self._n_train_steps_total % self.target_update_period == 0:
            self.qf.update_target(self.soft_target_tau)

        self.train_info.update(train_qf_info)

        if self._n_train_steps_total % self.policy_update_freq == 0:
            new_action, agent_info = self.policy.action(
                obs, 
                reparameterize=True, 
                return_log_prob=True,
            )
            if self.use_automatic_entropy_tuning:

                alpha_loss, train_alpha_info = self.compute_alpha_loss(agent_info)
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.train_info.update(train_alpha_info)

            policy_loss, train_policy_info = self.compute_policy_loss(
                obs, 
                new_action, 
                agent_info, 
                self.policy_v_pi_kwargs
            )

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.train_info.update(train_policy_info)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics.update(self.train_info)
            
        self._n_train_steps_total += 1
        return copy.deepcopy(self.train_info)



"""
dead code:

        if self.compute_target_mode == 'mn_min':
            m = self.compute_target_kwargs["m"] 
            n = self.compute_target_kwargs["n"] 
            indices = np.random.choice(ensemble_size, imagined_batch_size*m)
            n_arange = np.repeat(np.arange(imagined_batch_size), m)

            imagined_next_obs_mean = imagined_next_obs_mean[n_arange, indices]
            imagined_next_obs_std = imagined_next_obs_std[n_arange, indices]
            
            if 'ensemble_reward_mean' in imagined_batch:
                compute_reward = False
                imagined_reward_mean = imagined_batch['ensemble_reward_mean'][n_arange, indices]
                imagined_reward_std = imagined_batch['ensemble_reward_std'][n_arange, indices]
            imagined_target = q_target[n_real:]
            imagined_target = imagined_target.reshape(imagined_batch_size,m)
            imagined_target,_ = torch.sort(imagined_target, dim=-1) 
            imagined_target = imagined_target[...,:n].mean(dim=-1,keepdim=True)
        q_target = torch.cat([q_target[:n_real], imagined_target],0)
"""