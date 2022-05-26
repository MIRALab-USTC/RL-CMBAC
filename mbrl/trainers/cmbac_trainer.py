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

from itertools import combinations
from scipy.special import comb

from mbrl.models.pe_model import sample_from_gaussian
from mbrl.models.utils import square_wass2_gaussian

import time

class CMBACTrainer(SACTrainer):
    def __init__(self, 
            env, 
            model,
            policy,
            qf,
            use_model_elite_indices=True,
            model_sample_num=3,
            **sac_trainer_kwargs
        ):
        super().__init__(env, policy, qf, **sac_trainer_kwargs)
        self.model = model
        self.reward_f, self.done_f = env.get_reward_done_function()
        self.use_model_elite_indices = use_model_elite_indices
        self.model_sample_num = model_sample_num
        self.list_of_index = None

    def train(self, real_data, imagined_data):
        return self.train_from_numpy_batch(real_data, imagined_data)

    def train_from_numpy_batch(self, np_real_batch, np_imagined_data):
        np_real_batch = ptu.np_to_pytorch_batch(np_real_batch)
        np_imagined_data = ptu.np_to_pytorch_batch(np_imagined_data)
        return self.train_from_torch_batch(np_real_batch, np_imagined_data)

    def _compute_model_uncertainty(self, obs, actions, u_type='std'):
        next_o, r, d, info = self.model.step(obs, actions, return_info=True, return_distribution=True)
        if u_type == 'std':
            ensemble_next_obs_std = info['ensemble_next_obs_std']
            u = torch.norm(ensemble_next_obs_std, dim=-1)
            u = torch.mean(u, dim=-1)
        elif u_type == 'w_d':
            m1 = info['next_obs_mean']
            std1 = info['next_obs_std']
            m2 = info['ensemble_next_obs_mean']
            std2 = info['ensemble_next_obs_std']

            # if 'reward_mean' in info:
            #     r_m1 = info['reward_mean']
            #     r_std1 = info["reward_std"]
            #     m1 = np.concatenate([m1,r_m1],-1)
            #     std1 = np.concatenate([std1,r_std1],-1)
            #     r_m2 = info['ensemble_reward_mean']
            #     r_std2 = info['ensemble_reward_std']
            #     m2 = np.concatenate([m2,r_m2],-1)
            #     std2 = np.concatenate([std2,r_std2],-1)
            assert (m2.shape == std2.shape and m2.dim() == 3)
            d = square_wass2_gaussian(m1, std1, m2, std2)
            u = d.mean(-1)
        else:
            raise NotImplementedError

        return u 

    def compute_qf_loss(self, obs, actions, q_target, prefix='QF/'):
        qf_info = OrderedDict()

        _, value_info = self.qf.value(
            obs, 
            actions, 
            return_ensemble=True,
            only_return_ensemble=True
        )
        q_value_ensemble = value_info['ensemble_atoms'] # (q_ensemble_size, batch, model_ensemble_size)
        qf_loss = ((q_value_ensemble - q_target.detach()) ** 2).mean()
        
        """
        LOG FOR Q FUNCTION
        """
        qf_info[prefix+'loss'] = np.mean(ptu.get_numpy(qf_loss))

        if LOG_LEVEL >= EXTRA_INFO or self._need_to_update_eval_statistics:
            qf_extra_info = OrderedDict()
            qf_extra_info.update(create_stats_ordered_dict(
                    prefix+'target',
                    ptu.get_numpy(q_target),
                ))
            if LOG_LEVEL >= EXTRA_INFO:
                qf_info.update(qf_extra_info)
            else:
                self.eval_statistics.update(qf_extra_info)

            # add model uncertainty
            # TODO: add different uncertainty mode 
            u = self._compute_model_uncertainty(obs, actions)
            model_info = OrderedDict()
            model_info.update(create_stats_ordered_dict(
                        prefix+'model_pred_uncertainty',
                        ptu.get_numpy(u),
                    ))
            self.eval_statistics.update(model_info)

        if LOG_LEVEL >= FULL_INFO or self._need_to_update_eval_statistics:
            qf_full_info = OrderedDict()
            q_pred_mean = torch.mean(q_value_ensemble, dim=0)
            qf_full_info.update(create_stats_ordered_dict(
                prefix+'pred_mean',
                ptu.get_numpy(q_pred_mean),
            ))
            q_pred_std = torch.std(q_value_ensemble, dim=0)
            qf_full_info.update(create_stats_ordered_dict(
                prefix+'pred_std',
                ptu.get_numpy(q_pred_std),
            ))
            uncertainty = torch.std(q_value_ensemble, dim=-1)
            qf_full_info.update(create_stats_ordered_dict(
                prefix+'local_uncertainty',
                ptu.get_numpy(uncertainty),
            ))
            if q_value_ensemble.dim() > 2:
                n_model = q_value_ensemble.shape[-1]
                for i in range(n_model):
                    qf_full_info.update(create_stats_ordered_dict(
                            prefix+'target_%d'%i,
                            ptu.get_numpy(q_target[:,i]),
                        ))
                for i in range(n_model):
                    qf_full_info.update(create_stats_ordered_dict(
                            prefix+'pred_mean_%d'%i,
                            ptu.get_numpy(q_pred_mean[:,i]),
                        ))

            if LOG_LEVEL >= FULL_INFO:
                qf_info.update(qf_full_info)
            else:
                self.eval_statistics.update(qf_full_info)
                
        return qf_loss, qf_info

    def compute_q_target(self, next_obs, rewards, terminals, v_pi_kwargs={}):
        alpha = self._get_alpha()

        next_action, next_policy_info = self.policy.action(
            next_obs, reparameterize=False, return_log_prob=True,
        )
        log_prob_next_action = next_policy_info['log_prob']
        _, target_info = self.qf.value(
            next_obs, 
            next_action, 
            use_target_value=True, 
            return_ensemble=True,
            only_return_ensemble=True,
            **v_pi_kwargs
        ) 
        target_q_next_action = torch.diagonal(target_info["ensemble_atoms"], dim1=2, dim2=3)
        target_q_next_action = torch.unsqueeze(torch.min(target_q_next_action, dim=1)[0], dim=-1)
        target_q_next_action = target_q_next_action - alpha * log_prob_next_action

        q_target = self.reward_scale * rewards + \
            (1. - terminals) * self.discount * target_q_next_action
        return q_target

    def _get_pertumation_next_obs(self, imagined_next_obs, imagined_rewards, imagined_terminals):
        # t_s = time.time()
        # assert imagined_next_obs.shape[-2] == imagined_rewards.shape[-2]
        if self.list_of_index is None:
            self.list_of_index = np.arange(imagined_next_obs.shape[-2])
            self.ensemble_size = imagined_next_obs.shape[-2]
            self.repeat_num = int(comb(self.ensemble_size, self.model_sample_num))
        i = 0
        # final_index = []
        for indexes in combinations(self.list_of_index, self.model_sample_num):
            index = np.random.choice(indexes)
            # final_index.append(index)
            tmp_imagined_next_obs = imagined_next_obs[:,index,:]
            tmp_imagined_rewards = imagined_rewards[:,index,:]
            tmp_imagined_terminals = imagined_terminals[:,index,:]
            tmp_imagined_next_obs = torch.unsqueeze(tmp_imagined_next_obs,-2)
            tmp_imagined_rewards = torch.unsqueeze(tmp_imagined_rewards, -2)
            tmp_imagined_terminals = torch.unsqueeze(tmp_imagined_terminals,-2)

            if i == 0:
                next_obs = tmp_imagined_next_obs
                next_rewards = tmp_imagined_rewards
                next_terminals = tmp_imagined_terminals
            else:
                next_obs = torch.cat([next_obs, tmp_imagined_next_obs], -2)
                next_rewards = torch.cat([next_rewards, tmp_imagined_rewards], -2)
                next_terminals = torch.cat([next_terminals, tmp_imagined_terminals], -2)
            i += 1
        return next_obs, next_rewards, next_terminals

    def train_from_torch_batch(self, batch, imagined_batch):
        # if self.use_model_elite_indices:
        #     self.target_v_pi_kwargs['atom_fliter'] = self.model.elite_indices
        #     self.policy_v_pi_kwargs['atom_fliter'] = self.model.elite_indices

        real_obs = batch['observations']
        imagined_obs = imagined_batch['observations']
        obs = torch.cat([real_obs, imagined_obs], 0)

        n_real = real_obs.shape[0]

        real_actions = batch['actions']
        imagined_actions = imagined_batch['actions']
        actions = torch.cat([real_actions, imagined_actions], 0)

        real_next_obs = batch['next_observations']
        real_rewards = batch['rewards']
        real_terminals = batch['terminals']

        compute_reward = True
        compute_terminal = True
        
        #TODO: if resample
        if True:
            imagined_next_obs_mean = imagined_batch['ensemble_next_obs_mean']
            imagined_next_obs_std = imagined_batch['ensemble_next_obs_std']
            ensemble_size = imagined_next_obs_mean.shape[-2]
            obs_shape = imagined_next_obs_mean.shape[-1]
            if 'ensemble_reward_mean' in imagined_batch:
                compute_reward = False
                imagined_reward_mean = imagined_batch['ensemble_reward_mean']
                imagined_reward_std = imagined_batch['ensemble_reward_std']

        imagined_next_obs = ptu.normal(
            imagined_next_obs_mean, 
            imagined_next_obs_std
        )
        if compute_reward:
            raise NotImplementedError
        else:
            imagined_rewards = ptu.normal(
                imagined_reward_mean, 
                imagined_reward_std
            )
        if compute_terminal:
            imagined_terminals = self.done_f(
                imagined_obs[:,None,:], 
                imagined_actions[:,None,:], 
                imagined_next_obs
            )
        
        # TODO: merge with maven trainer
        if self.use_model_elite_indices:
            imagined_next_obs = imagined_next_obs[:,self.model.elite_indices,:]
            imagined_rewards = imagined_rewards[:,self.model.elite_indices,:]
            imagined_terminals = imagined_terminals[:,self.model.elite_indices,:]
        
        if self.model_sample_num > 1:
            imagined_next_obs, imagined_rewards, imagined_terminals = self._get_pertumation_next_obs(imagined_next_obs, \
                                            imagined_rewards, imagined_terminals)
        ensemble_size = imagined_next_obs.shape[-2]

        real_next_obs = torch.unsqueeze(real_next_obs,-2)
        real_rewards = torch.unsqueeze(real_rewards,-2)
        real_terminals = torch.unsqueeze(real_terminals,-2)

        real_next_obs = real_next_obs.repeat(1,ensemble_size,1)
        real_rewards = real_rewards.repeat(1,ensemble_size,1)
        real_terminals = real_terminals.repeat(1,ensemble_size,1)

        next_obs = torch.cat([real_next_obs, imagined_next_obs], 0)
        rewards = torch.cat([real_rewards, imagined_rewards], 0)
        terminals = torch.cat([real_terminals, imagined_terminals], 0)

        q_target = self.compute_q_target(
            next_obs, 
            rewards, 
            terminals, 
            self.target_v_pi_kwargs
        ) # (batchsize, model_ensemble_size, 1)
        q_target = torch.squeeze(q_target,-1) # (batchsize, model_ensemble_size, 1)

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