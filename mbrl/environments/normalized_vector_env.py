import random
import numpy as np
import gym
from gym.spaces import Box
from gym import Wrapper
from gym.vector import SyncVectorEnv, AsyncVectorEnv
import random
from mbrl.environments.base_env import Env
from mbrl.environments.utils import get_make_fns
from mbrl.environments.reward_done_functions import get_reward_done_function
from mbrl.utils.mean_std import RunningMeanStd
from collections import Iterable
import warnings

class NormalizedVectorEnv(Env):
    def __init__(self, 
                 env_name,
                 n_env=1, 
                 reward_scale=1.0,
                 max_length=np.inf,
                 should_normalize_obs=False, 
                 known=[],
                 asynchronous=True,
                 **vector_env_kwargs):
        super().__init__(env_name, known)
        self.n_env = n_env
        self.cur_seeds = [random.randint(0,65535) for i in range(n_env)]
        self.make_fns = get_make_fns(env_name, self.cur_seeds, n_env)
        if asynchronous:
            inner_env = AsyncVectorEnv(self.make_fns,**vector_env_kwargs)
        else:
            inner_env = SyncVectorEnv(self.make_fns,**vector_env_kwargs)
        Wrapper.__init__(self, inner_env)
        self.asynchronous = asynchronous

        self.reward_scale = reward_scale
        self.max_length = max_length
        self.low = np.maximum(self.env.single_action_space.low, -10)
        self.high = np.minimum(self.env.single_action_space.high, 10)
        self.observation_space = self.env.single_observation_space
        ub = np.ones(self.env.single_action_space.shape)
        self.action_space = Box(-1 * ub, ub)

        self.should_normalize_obs = should_normalize_obs
        if should_normalize_obs:
            self.obs_mean_std = RunningMeanStd(self.observation_space.shape)
        self.reset()
    
    def _normalize_observation(self, obs):
        return (obs - self.obs_mean_std.mean) / np.sqrt(self.obs_mean_std.var + 1e-12)

    def get_state(self):
        assert not self.asynchronous
        sim_data = []
        for e in self.env.envs:
            sim_data.append(e.sim.get_state())
        return sim_data

    def set_state(self, sim_data):
        assert not self.asynchronous
        if isinstance(sim_data,Iterable):
            for e,d in zip(self.env.envs, sim_data):
                e.sim.set_state(d)
        else:
            for e in self.env.envs:
                e.sim.set_state(sim_data)

    def state_vector(self):
        assert not self.asynchronous
        obs = []
        for e in self.env.envs:
            if hasattr(e, '_get_obs'):
                obs.append(e._get_obs())
            else:
                obs.append(e.state_vector())
        return np.stack(obs)
    
    @property
    def horizon(self):
        return self.max_length

    def reset(self):
        self.cur_step_id = 0
        obs = self.env.reset()
        if self.should_normalize_obs:
            self.obs_mean_std.update(obs)
            obs = self._normalize_observation(obs)
        return obs

    def step(self, action):
        self.cur_step_id = self.cur_step_id + 1
        action = np.clip(action, -1.0, 1.0)
        action = self.low + (action + 1.0) * (self.high - self.low) * 0.5
        if len(action.shape) == len(self.action_space.shape):
            action = np.stack([action] * self.n_env)
        o,r,d,infos=self.env.step(action)
        if self.should_normalize_obs:
            self.obs_mean_std.update(o)
            o = self._normalize_observation(o)
        r = r.reshape(self.n_env,1)
        d = d.reshape(self.n_env,1).astype(np.float)
        if self.cur_step_id >= self.max_length:
            done = np.ones((self.n_env,1), dtype=np.float)
        new_info = {}
        for k in infos[0]:
            new_info[k] = np.empty((self.n_env,1),dtype=np.float)
        for i,info in enumerate(infos):
            for k,v in info.items():
                new_info[k][i,0] = v
        return o, r, d, new_info
        
