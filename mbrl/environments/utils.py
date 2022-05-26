from mbrl.environments.our_envs import env_name_to_gym_registry_dict
from mbrl.collectors.utils import rollout
import gym
import warnings

def env_name_to_gym_registry(env_name):
    if env_name in env_name_to_gym_registry_dict:
        return env_name_to_gym_registry_dict[env_name]
    return env_name

def make_gym_env(env_name, seed):
    env = gym.make(env_name_to_gym_registry(env_name)).env
    env.seed(seed)
    return env

def get_make_fn(env_name, seed):
    def make():
        env = gym.make(env_name_to_gym_registry(env_name)).env
        env.seed(seed)
        return env
    return make

def get_make_fns(env_name, seeds, n_env=1):
    if seeds is None:
        seeds = [None] * n_env
    elif len(seeds) != n_env:
        warnings.warn('the length of the seeds is different from n_env')

    make_fns = [get_make_fn(env_name, seed) for seed in seeds]
    return make_fns
