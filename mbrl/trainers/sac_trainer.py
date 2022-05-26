from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn

import mbrl.torch_modules.utils as ptu
from mbrl.utils.eval_util import create_stats_ordered_dict
from mbrl.trainers.base_trainer import BatchTorchTrainer
from mbrl.global_config import *
import pdb
import copy

class SACTrainer(BatchTorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=3e-4,
            qf_lr=3e-4,
            policy_update_freq=1,
            optimizer_class='Adam',

            soft_target_tau=5e-3,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            alpha_if_not_automatic=1e-2,
            use_automatic_entropy_tuning=True,
            init_log_alpha=0,
            target_entropy=None,

            target_v_pi_kwargs={}, 
            policy_v_pi_kwargs={}, 
    ):
        super().__init__()
        if isinstance(optimizer_class, str):
            optimizer_class = eval('optim.'+optimizer_class)
        self.env = env
        self.policy = policy
        self.qf = qf
        self.policy_update_freq = policy_update_freq
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.alpha_if_not_automatic = alpha_if_not_automatic
        if self.use_automatic_entropy_tuning:
            if target_entropy is not None:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  
            self.log_alpha = ptu.FloatTensor([init_log_alpha])
            self.log_alpha.requires_grad_(True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.policy_optimizer = optimizer_class(
            self.policy.module.parameters(),
            lr=policy_lr,
        )
        if isinstance(qf, list):
            self.qf_optimizer = []
            for q in self.qf:
                qf_optimizer = optimizer_class(
                    q.module.parameters(),
                    lr=qf_lr,
                )
                self.qf_optimizer.append(qf_optimizer)
        else:
            self.qf_optimizer = optimizer_class(
                self.qf.module.parameters(),
                lr=qf_lr,
            )

        self.discount = discount
        self.reward_scale = reward_scale
        self.train_info = OrderedDict()
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.target_v_pi_kwargs = target_v_pi_kwargs
        self.policy_v_pi_kwargs = policy_v_pi_kwargs

    def _get_alpha(self):
        if self.use_automatic_entropy_tuning:
            alpha = self.log_alpha.exp() 
        else:
            alpha = self.alpha_if_not_automatic
        return alpha

    def compute_q_target(self, next_obs, rewards, terminals, v_pi_kwargs={}):
        alpha = self._get_alpha()

        next_action, next_policy_info = self.policy.action(
            next_obs, reparameterize=False, return_log_prob=True,
        )
        log_prob_next_action = next_policy_info['log_prob']
        target_q_next_action = self.qf.value(
            next_obs, 
            next_action, 
            use_target_value=True, 
            return_info=False,
            **v_pi_kwargs
        ) - alpha * log_prob_next_action

        q_target = self.reward_scale * rewards + \
            (1. - terminals) * self.discount * target_q_next_action
        return q_target

    def compute_qf_loss(self, obs, actions, q_target, prefix='QF/'):
        qf_info = OrderedDict()

        _, value_info = self.qf.value(obs, actions, return_ensemble=True)
        q_value_ensemble = value_info['ensemble_value']
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
            if LOG_LEVEL >= FULL_INFO:
                qf_info.update(qf_full_info)
            else:
                self.eval_statistics.update(qf_full_info)
                
        return qf_loss, qf_info

    def compute_alpha_loss(self, agent_info, prefix='alpha/'):
        alpha_info = OrderedDict()
        log_prob_new_action = agent_info['log_prob']
        alpha_loss = -(self.log_alpha * (log_prob_new_action + self.target_entropy).detach()).mean()

        # compute sttistics
        alpha = self.log_alpha.exp() 
        alpha_info[prefix+'value'] = alpha.item()
        alpha_info[prefix+'loss'] = alpha_loss.item()

        return alpha_loss, alpha_info

    def compute_policy_loss(self, obs, new_action, agent_info, v_pi_kwargs={}, prefix='policy/'):
        policy_info = OrderedDict()
        log_prob_new_action = agent_info['log_prob']
        alpha = self._get_alpha()

        q_new_action, _ = self.qf.value(
            obs, 
            new_action, 
            return_ensemble=False, 
            **v_pi_kwargs
        ) 
        
        entropy = -log_prob_new_action.mean()
        q_pi_mean = q_new_action.mean()
        policy_loss = -alpha*entropy - q_pi_mean

        policy_info[prefix+'loss'] = policy_loss.item()
        policy_info[prefix+'q_pi'] = q_pi_mean.item()
        policy_info[prefix+'entropy'] = entropy.item()

        return policy_loss, policy_info

    def train_from_torch_batch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        q_target = self.compute_q_target(next_obs, rewards, terminals, self.target_v_pi_kwargs)
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

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf=self.qf
        )
        
    def anomaly_detection(self):
        if DEBUG and self._n_train_steps_total>0 \
            and self.eval_statistics['QF Loss'] > 5e4:
            pdb.set_trace()

class SACMultiHeadQTrainer(SACTrainer):
    def compute_qf_loss(self, obs, actions, q_target, prefix='QF/'):
        qf_info = OrderedDict()

        _, value_info = self.qf.value(obs, actions, return_ensemble=True)
        q_value_ensemble = value_info['ensemble_atoms']
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

        if LOG_LEVEL >= FULL_INFO or self._need_to_update_eval_statistics:
            qf_full_info = OrderedDict()
            # q_pred_mean = torch.mean(q_value_ensemble, dim=0)
            # qf_full_info.update(create_stats_ordered_dict(
            #     prefix+'pred_mean',
            #     ptu.get_numpy(q_pred_mean),
            # ))
            # q_pred_std = torch.std(q_value_ensemble, dim=0)
            # qf_full_info.update(create_stats_ordered_dict(
            #     prefix+'pred_std',
            #     ptu.get_numpy(q_pred_std),
            # ))
            # if LOG_LEVEL >= FULL_INFO:
            #     qf_info.update(qf_full_info)
            # else:
            #     self.eval_statistics.update(qf_full_info)
                
        return qf_loss, qf_info