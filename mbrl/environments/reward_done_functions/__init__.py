import importlib.util
import os.path as osp
from mbrl.utils.misc_untils import to_list
from mbrl.environments.our_envs import env_name_to_gym_registry_dict, get_short_name

def get_reward_done_function(env_name, known=[]):
    env_name = get_short_name(env_name)
    known = to_list(known)
    if len(known) > 0:
        module_name = 'mbrl.environments.reward_done_functions.'+env_name
        module = importlib.import_module(module_name)

        if 'reward_function' in known:
            reward_function = module.reward_function
        else:
            reward_function = None
        if 'done_function' in known:
            done_function = module.done_function
        else:
            done_function = None

        return reward_function, done_function
    else:
        return None, None