import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from collections import OrderedDict

from mbrl.models.base_model import Model
from mbrl.torch_modules.mlp import MLP
from mbrl.utils.logger import logger
from mbrl.processors.base_processor import PairProcessor
import mbrl.torch_modules.utils as ptu
from mbrl.utils.eval_util import create_stats_ordered_dict

import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -20
LOG2PI = np.log(2*np.pi)

def sample_from_ensemble(input_tensor,
                         sample_number=1,
                         elite_indices=None,
                         replace=True):
    input_size = input_tensor.shape
    assert len(input_size) == 3
    ensemble_size = input_size[0]
    batch_size = input_size[1]
    shape = input_size[-1]

    if elite_indices is None:
        elite_indices = ensemble_size
    
    if replace or (sample_number == 1):
        indices = np.random.choice(elite_indices, batch_size*sample_number)
    else:
        assert sample_number <= ensemble_size
        
        indices = [np.random.choice(ensemble_size-i, (batch_size,)) for i in range(sample_number)]
        indices = np.stack(indices)
        for i in range(sample_number-1,0,-1):
            for j in range(i,sample_number):
                indices[j] = (indices[j] + indices[i-1] + 1) % (ensemble_size + 1 -i)
        indices = indices.flatten()

    n_arange = np.tile(np.arange(batch_size), sample_number)

    output_tensor = input_tensor[indices, n_arange]
    output_tensor = output_tensor.reshape(sample_number, batch_size, shape)
    return output_tensor, indices

def sample_from_gaussian(mean,
                       std,
                       reparameterize=True):
    assert mean.dim() == 3 and mean.shape == std.shape

    #sample from Gaussian
    if reparameterize:
        samples = (
                    mean +
                    std *
                    Normal(
                        ptu.zeros_like(mean),
                        ptu.ones_like(mean)
                    ).sample()
                )
        samples.requires_grad_(True)
    else:
        samples = ptu.normal(mean, std)

    return samples

class PEModelModule(MLP):
    def __init__(self,
                 obs_size, 
                 action_size, 
                 learn_reward=True,
                 learn_done=False,
                 ensemble_size=None,
                 model_name='PE', 
                 learn_std=True,
                 **mlp_kwargs):
        self.obs_size = obs_size
        self.action_size = action_size
        self.learn_reward = learn_reward
        self.learn_done = learn_done
        if self.learn_reward and self.learn_done:
            output_size = (obs_size+1)*2+1 #mean_obs, mean_r, std_obs, std_r, term_prob
            self.mean_std_size = obs_size+1
        elif self.learn_reward:
            output_size = (obs_size+1)*2 #mean_obs, mean_r, std_obs, std_r, 
            self.mean_std_size = obs_size+1
        elif (not self.learn_reward) and (not self.learn_done):
            output_size = obs_size * 2 #mean_obs, std_obs
            self.mean_std_size = obs_size
        else:
            raise NotImplementedError
        
        self.obs_size = obs_size
        self.action_size = action_size

        super(PEModelModule, self).__init__(
            obs_size+action_size,
            output_size, 
            ensemble_size=ensemble_size,
            module_name=model_name,
            **mlp_kwargs
        )

        self.max_log_std = nn.Parameter(ptu.ones(1, self.mean_std_size, dtype=torch.float32) / 4.0, requires_grad=learn_std)
        self.min_log_std = nn.Parameter(-ptu.ones(1, self.mean_std_size, dtype=torch.float32) * 5.0, requires_grad=learn_std)

    def get_extra_loss(self, coefficient=0.05):
        diagnostics = OrderedDict()
        diagnostics.update(
            create_stats_ordered_dict(
                "max_log_std",
                ptu.get_numpy(self.max_log_std),
            )
        )
        diagnostics.update(
            create_stats_ordered_dict(
                "min_log_std",
                ptu.get_numpy(self.min_log_std),
            )
        )

        # return coefficient * (torch.sum(self.max_log_std)  - torch.sum(self.min_log_std)), diagnostics
        return coefficient * (torch.mean(self.max_log_std)  - torch.mean(self.min_log_std)), diagnostics

    def _get_mean_logstd(self, tensor):
        size = self.mean_std_size
        mean = tensor[...,:size]
        log_std = tensor[...,size:2*size]
        log_std = self.max_log_std - F.softplus(self.max_log_std - log_std)
        log_std = self.min_log_std + F.softplus(log_std - self.min_log_std)
        return mean, log_std


    def log_prob(self,
                 obs,
                 action,
                 delta,
                 reward=None,
                 done=None,
                 epsilon=1e-6):
        assert obs.dim() == action.dim()
        input_tensor = torch.cat([obs, action], dim=-1)
        output = super().forward(input_tensor)
        assert output.dim() == 3
        
        mean, log_std = self._get_mean_logstd(output)

        if self.learn_reward:
            assert reward is not None
            target = torch.cat([delta, reward], -1)
        else:
            target = delta

        if self.learn_done:
            assert done is not None
        
        log_var = log_std * 2
        inv_var = torch.exp(-log_var)
        mse_loss = (target - mean) ** 2
        # mse_loss = F.mse_loss(mean, target, reduction="none")
        log_prob = -0.5 *  (LOG2PI + log_var + mse_loss * inv_var)
        # log_prob = -1.0 *  (log_var + mse_loss * inv_var)
        if self.learn_done:
            p = torch.sigmoid(output[...,-1:])
            log_prob2 = done * torch.log(p+epsilon) + (1-done) * torch.log(1-p+epsilon)
            log_prob = torch.cat([log_prob, log_prob2], -1)

        return log_prob, log_std

    def forward(
        self, 
        obs, 
        action,
        return_info=True,
        sample_number=1,
        reparameterize=True,
        elite_indices=None,
        return_all_samples=False, 
        return_distribution=False,
        return_indices=True,
    ):
        # NOTE: we predict delta (next_obs - obs) instead of next_obs
        assert obs.dim() == action.dim()
        input_tensor = torch.cat([obs, action], dim=-1)
        output = super().forward(input_tensor)
        assert output.dim() == 3

        if elite_indices is None:
            elite_indices = list(np.arange(self.ensemble_size))
        all_samples, indices = sample_from_ensemble(output, sample_number, elite_indices)

        mean, log_std = self._get_mean_logstd(all_samples)
        std = torch.exp(log_std)
        all_deltas_rewards = sample_from_gaussian(mean,
                                                   std,
                                                   reparameterize)
        all_deltas = all_deltas_rewards[...,:self.obs_size]
        delta = all_deltas[0]
        if self.learn_reward:
            all_rewards = all_deltas_rewards[...,self.obs_size:]
            reward = all_rewards[0]
        else:
            reward = None

        if self.learn_done:
            all_dones = torch.sigmoid(all_samples[...,-1:])
            done = all_dones[0]
        else:
            done = None

        if return_info:
            info = {}
            if return_indices:
                info["sample_indices"] = torch.IntTensor(indices)
                # ptu.from_numpy(indices)
            if return_all_samples:
                info['all_sampled_delta'] = all_deltas.transpose(0,1)
                if self.learn_reward:
                    info['all_sampled_reward'] = all_rewards.transpose(0,1)


            if return_distribution:
                output = output.transpose(0,1)
                ensemble_mean, ensemble_log_std = self._get_mean_logstd(output)
                ensemble_std = torch.exp(ensemble_log_std)

                info['delta_mean'] = mean[0][...,:self.obs_size]
                info['delta_std'] = std[0][...,:self.obs_size]
                info['ensemble_delta_mean'] = ensemble_mean[...,:self.obs_size]
                info['ensemble_delta_std'] = ensemble_std[...,:self.obs_size]

                if self.learn_reward:
                    info['reward_mean'] = mean[0][...,-1:]
                    info['reward_std'] = std[0][...,-1:]
                    info['ensemble_reward_mean'] = ensemble_mean[...,-1:]
                    info['ensemble_reward_std'] = ensemble_std[...,-1:]

            return delta, reward, done, info
        else:
            return delta, reward, done

    def process(self, obs, action, embedding_layer=-2):
        assert obs.dim() == action.dim()
        input_tensor = torch.cat([obs, action], dim=-1)
        output = super().forward(input_tensor, embedding_layer)
        assert output.dim() == 3
        return output


class PEModel(nn.Module, Model, PairProcessor):
    def __init__(self,  
                 env,
                 normalize_obs=True,
                 normalize_action=True,
                 normalize_delta=True,
                 normalize_reward=False,
                 known=None,
                 ensemble_size=7,
                 elite_number=5,
                 embedding_layer=-2,
                 model_name='PE_model',
                 **mlp_kwargs):
        nn.Module.__init__(self)
        Model.__init__(self,  
                       env,
                       normalize_obs,
                       normalize_action,
                       normalize_delta,
                       normalize_reward,
                       known)

        assert len(self.processed_obs_shape) == 1 and len(self.processed_action_shape) == 1
        self.ensemble_size = ensemble_size
        self.elite_number = elite_number

        self.module = PEModelModule(self.processed_obs_shape[0], 
                                    self.processed_action_shape[0],
                                    self.learn_reward,
                                    self.learn_done,
                                    self.ensemble_size,
                                    model_name,
                                    **mlp_kwargs)
        self.elite_indices = None
        self.rank = list(np.arange(ensemble_size))
        self.loss = None

        self.embedding_layer = embedding_layer
        self.output_shape = (self.module.final_layers[embedding_layer],)
        self.embedding_coef = nn.Parameter(ptu.zeros(ensemble_size, 1, 1, dtype=torch.float32), requires_grad=True)
    
    def remember_loss(self, loss):
        self.loss = loss
        self.rank = np.argsort(loss)
        self.elite_indices = list(self.rank[:self.elite_number])

    def _predict_delta (self, 
                obs, 
                action,
                return_info=True,
                sample_number=1,
                reparameterize=True,
                return_all_samples=False, 
                return_distribution=False):
        return self.module(obs, 
                        action,
                        return_info,
                        sample_number,
                        reparameterize,
                        self.elite_indices,
                        return_all_samples, 
                        return_distribution)

    
    def step(self, obs, action, return_info=True, **kwargs):
        if return_info:
            next_obs, r, d, info = super().step(obs, action, True, **kwargs)
            # NOTE: recover by normalizer
            # get [all_sampled_/ensemble_]next_obs[_mean/_std]
            keys = list(info.keys())
            for k in keys:
                v = info[k]
                if self.normalize_obs and 'delta' in k:
                    new_k = k.replace('delta', 'next_obs')
                    if 'sampled' in k or 'mean' in k:
                        v = self.delta_processor.recover(v)
                        info[k] = v
                        if "ensemble" in k:
                            info[new_k] = v + obs[:,None,:]
                        else:
                            info[new_k] = v + obs
                    if 'std' in k:
                        std, epsilon = self.delta_processor.std, self.delta_processor.epsilon
                        v = v*(std+epsilon)
                        info[k] = v
                        info[new_k] = v

                if self.learn_reward and self.normalize_reward and 'reward' in k:
                    if 'sampled' in k or 'mean' in k:
                        info[k] = self.reward_processor.recover(v)
                    if 'std' in k:
                        std, epsilon = self.reward_processor.std, self.reward_processor.epsilon
                        info[k] = v*(std+epsilon)
            return next_obs, r, d, info
        else:
            return super().step(obs, action, False, **kwargs)

    def log_prob(self,
                 obs,
                 action,
                 delta,
                 reward=None,
                 done=None):
        if self.normalize_obs:
            obs = self.obs_processor.process(obs)
        if self.normalize_action:
            action = self.action_processor.process(action)
        if self.normalize_delta:
            delta = self.delta_processor.process(delta)
        if self.learn_reward and self.normalize_reward:
            reward = self.reward_processor.process(reward)

        return self.module.log_prob(obs, action, delta, reward, done)

    def process(self, obs, action, without_gradient=True):
        if self.normalize_obs:
            obs = self.obs_processor.process(obs)
        if self.normalize_action:
            action = self.action_processor.process(action)
        output = self.module.process(obs, action, self.embedding_layer)
        if without_gradient:
            output = output.detach()
        output = output * torch.softmax(self.embedding_coef, 0)
        output = torch.sum(output, 0)
        return output
        
    def log_prob_np(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, save_dir=None, net_id=None):
        if save_dir == None:
            save_dir = logger._snapshot_dir
        self.module.save(save_dir, net_id)
    
    def load(self, load_dir=None, net_id=None):
        if load_dir == None:
            load_dir = logger._snapshot_dir
        self.module.load(load_dir, net_id)

    def get_snapshot(self):
        return self.module.get_snapshot()

