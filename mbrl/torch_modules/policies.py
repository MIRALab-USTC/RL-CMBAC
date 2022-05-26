import numpy as np
import abc
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import warnings

from mbrl.torch_modules.flows import RealNVP, RadialNormalizingFlow
from mbrl.torch_modules.mlp import MLP, NoisyMLP
from mbrl.torch_modules.linear import Linear
import mbrl.torch_modules.utils as ptu
from mbrl.torch_modules.linear import *
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class GaussianPolicyModule(MLP, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_mean_std(self, obs, return_log_std=False):
        pass

    def forward(
            self,
            obs,
            return_info=True,
            deterministic=False,
            reparameterize=True,
            return_log_prob=False,
            return_mean_std=False,
            return_entropy=False,
    ):
        """
        :param obs: Observation
        :param deterministic: 
        :param return_log_prob: 
        :return: 
        """
        mean, log_std = self.get_mean_std(obs,return_log_std=True)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            if reparameterize:
                action = (
                    mean +
                    std *
                    Normal(
                        ptu.zeros_like(mean),
                        ptu.ones_like(mean)
                    ).sample()
                )
                action.requires_grad_(True)
            else:
                action = normal.sample()

        if return_info:
            info = {}
            if return_log_prob:
                log_prob = normal.log_prob(action)
                info['log_prob'] = log_prob.sum(-1, keepdim=True)
            if return_mean_std:
                info['mean'] = mean
                info['std'] = std
            if return_entropy:
                a = 0.5 + 0.5 * math.log(2 * math.pi)
                entropy = torch.sum(a + log_std, dim=-1, keepdim=True)
                info['entropy'] = entropy
            return action, info
        else:
            return action

    def log_prob(self, obs, action):
        mean, std = self.get_mean_std(obs)
        normal = Normal(mean, std)
        log_prob = normal.log_prob(action)
        return log_prob.sum(-1, keepdim=True)

    def get_entroy(self, obs):
        _,log_std = self.get_mean_std(obs,return_log_std=True)
        a = 0.5 + 0.5 * math.log(2 * math.pi)
        entropy = torch.sum(a + log_std, dim=-1, keepdim=True)
        return entropy


class SimpleGaussianPolicyModule(GaussianPolicyModule):
    def __init__(self, 
                 obs_size, 
                 action_size, 
                 policy_name='simple_gaussian_policy',
                 **mlp_kwargs
                 ):
        super(SimpleGaussianPolicyModule, self).__init__(
            obs_size,
            action_size,
            module_name=policy_name,
            **mlp_kwargs
        )
        self.obs_size = obs_size
        self.action_size = action_size
        self.log_std = nn.Parameter(ptu.zeros(1,self.layers[-1]))

    def get_mean_std(self, obs, return_log_std=False):
        mean=MLP.forward(self, obs)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std = log_std.expand(mean.shape)
        if return_log_std:
            return mean, log_std
        else:
            return mean, torch.exp(log_std)

    def get_entroy_scalar(self):
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        a = 0.5 + 0.5 * math.log(2 * math.pi)
        entropy = torch.sum(a + log_std)
        return entropy

# the following is much faster than the "simple" one......
class MeanLogstdGaussianPolicyModule(GaussianPolicyModule):
    def __init__(self, 
                 obs_size, 
                 action_size, 
                 policy_name='mean_logstd_gaussian_policy',
                 **mlp_kwargs):
        super(MeanLogstdGaussianPolicyModule, self).__init__(
            obs_size,
            action_size*2,
            module_name=policy_name,
            **mlp_kwargs
        )
        self.obs_size = obs_size
        self.action_size = action_size

    def get_mean_std(self, obs, return_log_std=False):
        output=MLP.forward(self, obs)
        mean, log_std = torch.chunk(output,2,-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        if return_log_std:
            return mean, log_std
        else:
            return mean, torch.exp(log_std)

class NoisyNetworkPolicyModule(NoisyMLP):
    def __init__(self, 
                 obs_size, 
                 action_size, 
                 policy_name='noisy_network_policy',
                 **noisy_mlp_kwargs):
        super(NoisyNetworkPolicyModule, self).__init__(
            obs_size,
            action_size,
            module_name=policy_name,
            **noisy_mlp_kwargs
        )
        self.obs_size = obs_size
        self.action_size = action_size

    def forward(self, 
                obs,
                return_info=True,
                return_log_prob=False,
                reparameterize=True,
                **kwargs):
        action = super(NoisyNetworkPolicyModule, self).forward(obs, **kwargs)
        if not reparameterize:
            warnings.warn('set reparameterize False while using noisy network policy')
        if return_info:
            if return_log_prob:
                warnings.warn('require log_prob while using noisy network policy')
            info = {}
            return action, info
        else:
            return action

class MultiHeadPolicyModule(MLP):
    def __init__(self, 
                 obs_size, 
                 action_size, 
                 number_of_heads,
                 independent,
                 policy_name='noisy_network_policy',
                 **mlp_kwargs):
        super(MultiHeadPolicyModule, self).__init__(
            obs_size,
            action_size*number_of_heads,
            module_name=policy_name,
            **mlp_kwargs
        )
        self.obs_size = obs_size
        self.action_size = action_size
        self.number_of_heads = number_of_heads
        self.independent = independent

    def forward(self, 
                obs,
                return_info=True,
                deterministic=False,
                return_distribution=False,
                without_sampling=False,
                reparameterize=True,
                return_log_prob=False,
                **kwargs):
        actions = super(MultiHeadPolicyModule, self).forward(obs, **kwargs)

        batch_size = actions.shape[:-1]
        actions = actions.reshape(batch_size+(self.number_of_heads, self.action_size))

        if without_sampling:
            action = actions[...,0,:]
        else:
            ind_size = batch_size + (1,self.action_size)
            if deterministic:
                action = actions.mean(dim=-2)
            else:
                if self.independent:
                    ind = ptu.randint(0,self.number_of_heads,size=ind_size).long()
                else:
                    ind = ptu.randint(0,self.number_of_heads,size=batch_size + (1,1)).long()
                    ind = ind.expand(ind_size)
                action = actions.gather(-2, ind).squeeze(-2)

        if not reparameterize:
            warnings.warn('set reparameterize False while using multi-head policy')
        if return_log_prob:
            warnings.warn('require log_prob while using multi-head policy')

        if return_info:
            info = {}

            if return_distribution:
                info['actions'] = actions
            return action, info
        else:
            return action

class GeneratorPolicyModule(MeanLogstdGaussianPolicyModule):  
    def __init__(self, 
                 obs_size, 
                 action_size, 
                 last_hidden_each_dim=16,
                 last_nonlinearity='relu',
                 policy_name='generator_policy',
                 **mlp_kwargs):
        super(GeneratorPolicyModule, self).__init__(
            obs_size,
            action_size*last_hidden_each_dim,
            policy_name=policy_name,
            **mlp_kwargs
        )
        self.action_size=action_size
        self.last_hidden_each_dim = last_hidden_each_dim
        self.last_fc=Linear(last_hidden_each_dim, 1)
        self.last_nl=ptu.get_nonlinearity(last_nonlinearity)

    def forward(self, 
                obs, 
                return_info=True,
                deterministic=False,
                reparameterize=True,
                return_log_prob=False):
        h = super(GeneratorPolicyModule, self).forward(
            obs,
            False,
            deterministic,
            reparameterize
        )
        batch_size = h.shape[:-1]
        h = h.reshape(batch_size+(self.action_size,-1))
        actions = self.last_fc(h).squeeze(-1)

        if return_log_prob:
            warnings.warn('require log_prob while using generator policy')

        if return_info:
            return actions, {}
        else:
            return actions


class NormalizingFlowPolicyModule(MeanLogstdGaussianPolicyModule):  
    def __init__(self, 
                 obs_size, 
                 action_size,  
                 n_blocks=4, 
                 nf_type='radial',
                 nf_kwargs={},
                 policy_name='generator_policy',
                 **mlp_kwargs):
        super(NormalizingFlowPolicyModule, self).__init__(
            obs_size,
            action_size,
            policy_name=policy_name,
            **mlp_kwargs
        )
        if nf_type == 'nvp':
            self.NL = RealNVP(self.action_size, n_blocks, **nf_kwargs)
        elif nf_type == 'radial':
            self.NL = RadialNormalizingFlow(self.action_size, n_blocks, **nf_kwargs)

    def forward(self, 
                obs, 
                return_info=True,
                deterministic=False,
                reparameterize=True,
                return_log_prob=False):
        h, info = super(NormalizingFlowPolicyModule, self).forward(
            obs,
            True,
            deterministic,
            reparameterize,
            return_log_prob
        )
        actions, log_det = self.NL(h,return_log_prob)
        
        if return_log_prob:
            info['log_prob'] = info['log_prob'] - log_det
        if return_info:
            return actions, info
        else:
            return actions

    def log_prob(self, obs, action):
        raise NotImplementedError

# assert the PDF of the output distribution is continuous to ensure the correctness of log_prob
# otherwise, set 'discrete' as True
class TanhPolicyModule(nn.Module):
    def __init__(self, policy, discrete=False):
        super(TanhPolicyModule, self).__init__()
        self._inner_policy = policy
        self.discrete = discrete
    
    def forward(
            self,
            obs,
            return_info=True,
            return_pretanh_action=False,
            **kwargs
    ): 
        if return_info:
            pre_action, info = self._inner_policy(obs, return_info=True, **kwargs)
            action = torch.tanh(pre_action)
            keys = list(info.keys())
            for k in keys:
                v = info.pop(k)
                info['pretanh_'+k] = v
                if k == 'log_prob':
                    if self.discrete:
                        log_prob = v
                    else:
                        log_prob = v - torch.log(1 - action * action + 1e-6).sum(-1, keepdim=True)
                    info['log_prob'] = log_prob
            if return_pretanh_action:
                info['pretanh_action'] = pre_action
            return action, info
        else:
            pre_action = self._inner_policy(obs, return_info=False, **kwargs)
            action = torch.tanh(pre_action) 
            return action

    def log_prob(self, obs, action=None, pretanh_action=None, **kwargs):
        if pretanh_action is None:
            assert action is not None
            pretanh_action = torch.log((1+action)/(1-action) +1e-6) / 2
        else:
            assert pretanh_action is not None
            action = torch.tanh(pretanh_action)
        pre_log_prob = self._inner_policy.log_prob(obs, pretanh_action, **kwargs)
        log_prob = pre_log_prob - torch.log(1 - action * action + 1e-6).sum(-1, keepdim=True)
        return log_prob

    def save(self, save_dir, **kwargs):
        self._inner_policy.save(save_dir, **kwargs)

    def load(self, load_dir, **kwargs):
        self._inner_policy.load(load_dir, **kwargs)

    def get_snapshot(self, **kwargs):
        return self._inner_policy.get_snapshot(**kwargs)

    def load_snapshot(self, **kwargs):
        self._inner_policy.load_snapshot(**kwargs)
