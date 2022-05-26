import numpy as np
from contextlib import contextmanager
from collections import OrderedDict
import abc

import mbrl.torch_modules.utils as ptu
from torch.distributions import Normal


class Policy(object, metaclass=abc.ABCMeta):
    def __init__(self, env):
        self.action_shape = env.action_space.shape
        self.observation_shape = env.observation_space.shape
        
    @abc.abstractmethod
    def action(self, obs, return_info):
        """Compute actions given processed observations"""
        pass

    def action_np(self, obs, return_info=True, **kwargs):
        obs = ptu.from_numpy(obs)
        if return_info:
            action, info = self.action(obs, return_info=return_info, **kwargs)
            action = ptu.get_numpy(action)
            info = ptu.torch_to_np_info(info)
            return action, info
        else:
            return ptu.get_numpy(self.action(obs, return_info=return_info, **kwargs))

    def reset(self):
        pass 

    def save(self, save_dir=None):
        pass
    
    def load(self, load_dir=None):
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

class RandomPolicy(Policy, metaclass=abc.ABCMeta):
    def __init__(self, env, deterministic=False):
        super(RandomPolicy, self).__init__(env)
        self._deterministic = deterministic

    @contextmanager
    def set_deterministic(self, deterministic=True):
        """Context manager for changing the determinism of the policy.
        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
                to during the context. The value will be reset back to the
                previous value when the context exits.
        """
        was_deterministic = self._deterministic
        self._deterministic = deterministic
        yield
        self._deterministic = was_deterministic

    def log_prob(self, obs, action):
        raise NotImplementedError

    def log_prob_np(self, obs, action, **kwargs):
        obs = ptu.from_numpy(obs)
        action = ptu.from_numpy(action)
        return ptu.get_numpy(self.log_prob(obs, action, **kwargs))


class UniformlyRandomPolicy(Policy):
    def action(self, obs, return_info):
        shape = (obs.shape[0], *self.action_shape)
        action = ptu.rand(shape)*2-1
        if return_info:
            return action, {}
        else:
            return action

    def action_np(self, obs, return_log_prob=False, return_info=True):
        shape = (obs.shape[0], *self.action_shape)
        action = np.random.uniform(-1,1,shape)
        if return_info:
            return action, {}
        else:
            return action

    def reset(self):
        pass

