import random
import abc
import numpy as np
from gym import Wrapper
from gym.spaces import Box
from mbrl.environments.utils import make_gym_env
from mbrl.environments.reward_done_functions import get_reward_done_function
class Env(Wrapper, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, env_name, known=[]):
        self.env_name = env_name
        self.known = known
        self.reward_f, self.done_f = self.get_reward_done_function(known)
    
    @abc.abstractproperty
    def horizon(self):
        pass

    def get_reward_done_function(self, known=None):
        if known is None:
            known = self.known
        return get_reward_done_function(self.env_name, known)


class SimpleEnv(Env):
    def __init__(self, 
                 env_name,
                 noise_scale=None,
                 reward_scale=1.0,
                 max_length=np.inf,
                 known=[]):
                 
        super().__init__(env_name, known)
        self.cur_seed = random.randint(0,65535)
        inner_env = make_gym_env(env_name, self.cur_seed)
        Wrapper.__init__(self, inner_env)
        self.reward_scale = reward_scale
        self.max_length = max_length
        self.low = np.maximum(self.env.action_space.low, -10)
        self.high = np.minimum(self.env.action_space.high, 10)
        ub = np.ones(self.env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

        self.noise_scale = noise_scale
    

    @property
    def horizon(self):
        return self.max_length
    
    def reset(self):
        self.cur_step_id = 0
        return np.array([self.env.reset()])

    def step(self, action):
        self.cur_step_id = self.cur_step_id + 1
        action = action[0]
        if self.noise_scale is not None:
            noise = np.random.randn(*action.shape) * self.noise_scale
            action = action + noise
        action = np.clip(action, -1.0, 1.0)
        action = self.low + (action + 1.0) * (self.high - self.low) * 0.5
        o, r, d, info = self.env.step(action)
        if self.cur_step_id >= self.max_length:
            done = 1.0
        o, r, d = np.array([o]), np.array([[r]]), np.array([[d]])
        for k in info:
            info[k] = np.array([[info[k]]])
        return o, r, d, info
    