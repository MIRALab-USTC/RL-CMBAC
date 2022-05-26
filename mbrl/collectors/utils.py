import numpy as np
from mbrl.utils.misc_untils import combine_item
from collections import OrderedDict
from tqdm import tqdm

def rollout(
        env,
        agent,
        max_path_length=np.inf,
        aggregate=False, 
        return_length=False,
        stop_if_terminal=False,
        render=False,
        render_kwargs=None,
        use_tqdm=False,
    ):
    """
    You should be very careful when you set aggregate as True
    
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    o = env.reset()
    agent.reset()
    pb = PathBuilder(len(o))
    if render: 
        env.render(**render_kwargs)
    if use_tqdm:
        iterator = tqdm(range(max_path_length))
    else:
        iterator = range(max_path_length)
    for _ in iterator:
        a, agent_info = agent.action_np(o)
        next_o, r, d, env_info = env.step(a)
        t = pb.update(o,a,r,d,agent_info,env_info)
        o = next_o
        if render:
            env.render(**render_kwargs)
        if np.all(t) and stop_if_terminal:
            break
    return pb.finalize(next_o, aggregate, return_length)

class PathBuilder():
    def __init__(self, n_env):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminals = []
        self.agent_infos = {}
        self.env_infos = {}
        self.path_lens = np.zeros((n_env,1), dtype=np.int)
        self.t = np.full((n_env,1), False, dtype=np.bool)
        self.step_id = 0

    def update(self, o, a, r, d, agent_info, env_info):
        self.path_lens = self.path_lens + (1-self.t.astype(int))
        self.t = np.logical_or(self.t, d)
        self.observations.append(o)
        self.actions.append(a)
        self.rewards.append(r)
        self.terminals.append(self.t)
        if self.step_id == 0:
            for k in agent_info:
                self.agent_infos[k] = []
            for k in env_info:
                self.env_infos[k] = []
        for k in agent_info:
            self.agent_infos[k].append(agent_info[k])
        for k in env_info:
            self.env_infos[k].append(env_info[k])
        self.step_id += 1
        return self.t
    
    def get_terminal(self):
        return self.t

    def get_path_lens(self):
        return self.path_lens

    def finalize(self, next_o, aggregate, return_length):
        observations = np.array(self.observations)
        next_observations = np.vstack((observations[1:, :],np.expand_dims(next_o, 0)))
        for k in self.agent_infos:
            self.agent_infos[k] = np.array(self.agent_infos[k])
        for k in self.env_infos:
            self.env_infos[k] = np.array(self.env_infos[k])
        self.paths = dict(
            observations=observations,
            actions=np.array(self.actions),
            rewards=np.array(self.rewards),
            next_observations=next_observations,
            terminals=np.array(self.terminals),
            agent_infos=self.agent_infos,
            env_infos=self.env_infos,
        )
        self.path_lens = np.reshape(self.path_lens, (-1))
        if not aggregate:
            self.paths = split_paths(self.paths)
        if return_length:
            return self.paths, self.path_lens
        else:
            return self.paths

def get_single_path_info(info, index):
    single_path_info = {}
    for k in info:
        single_path_info[k] = info[k][:,index,:]
    return single_path_info

def split_paths(paths):
    new_paths = []
    for i in range(len(paths['actions'][0])):
        path = dict(
            observations=paths['observations'][:,i,:],
            actions=paths['actions'][:,i,:],
            rewards=paths['rewards'][:,i,:],
            next_observations=paths['next_observations'][:,i,:],
            terminals=paths['terminals'][:,i,:],
            agent_infos=get_single_path_info(paths['agent_infos'],i),
            env_infos=get_single_path_info(paths['env_infos'],i),
        )
        new_paths.append(path)
    return new_paths

def cut_path(path, target_length):
    new_path = {}
    for key, value in path.items():
        if type(value) in [dict, OrderedDict]:
            new_path[key] = cut_path(value, target_length)
        else:
            new_path[key] = value[:target_length]
    return new_path

def path_to_samples(paths):
    path_number = len(paths)
    data = paths[0]
    for i in range(1,path_number):
        data = combine_item(data, paths[i])
    return data



 