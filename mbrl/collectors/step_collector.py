from collections import deque, OrderedDict
import numpy as np
import torch

from mbrl.utils.eval_util import create_stats_ordered_dict
from mbrl.utils.misc_untils import combine_items, get_valid_part, split_items
from mbrl.collectors.base_collector import StepCollector
from mbrl.collectors.utils import PathBuilder, cut_path
import mbrl.torch_modules.utils as ptu
from contextlib import contextmanager



class SimpleStepCollector(StepCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._epoch_path_lens = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._epoch_num = 0
        self.has_done = True
        self.wipe_memory()
    
    def set_policy(self, policy, epoch=None):
        self.end_epoch(epoch)
        self._policy = policy

    @contextmanager
    def with_policy(self, policy):
        old_policy = self._policy
        self.set_policy(policy)
        yield
        self.set_policy(old_policy)

    def wipe_memory(self):
        self._num_steps_collected = 0
        self._res_data = None

    def _update_path_process_kwargs(
        self,
        max_path_length,
        discard_incomplete_paths,
        cut,
        stop_if_terminal,
    ):
        max_path_length = min(self._env.horizon,max_path_length)
        if self.has_done:
            self.max_path_length = max_path_length
            self.discard_incomplete_paths = discard_incomplete_paths
            self.cut = cut
            self.stop_if_terminal = stop_if_terminal
        else:
            assert self.max_path_length == max_path_length
            assert self.discard_incomplete_paths == discard_incomplete_paths
            assert self.cut == cut
            assert self.stop_if_terminal == stop_if_terminal

    def _get_valid_data(self, data):
        valid_data = [get_valid_part(item, data[-1]) for item in data]
        return tuple(valid_data)

    def collect_new_steps(
            self,
            num_steps,
            max_path_length=1000,
            discard_incomplete_paths=True,
            cut=True,
            stop_if_terminal=True,
            finalize=False,
    ):
        self._update_path_process_kwargs(max_path_length, discard_incomplete_paths, cut, stop_if_terminal)
        data = self._res_data
        while self._num_steps_collected < num_steps:
            if self._num_steps_collected == 0:
                cur_data = self._collect_one_step()
                data = cur_data
            else:
                cur_data = self._collect_one_step()
                data = combine_items(data, cur_data)
            self._num_steps_collected += np.sum(cur_data[-1])
            self._num_steps_total += np.sum(cur_data[-1])
        data = self._get_valid_data(data)

        if self._num_steps_collected > num_steps:
            data, self._res_data = split_items(data, num_steps)
            self._num_steps_collected -= num_steps
        else:
            self.wipe_memory()
        if finalize:
            self._handle_rollout_ending()
        NAME_LIST = ['observations', 'actions', 'next_observations', 'rewards', 'terminals', 'agent_infos', 'env_infos']
        return {k:v for (k,v) in zip(NAME_LIST, data[:-1])}

    def _collect_one_step(self):
        if self.has_done:
            self._start_new_rollout()
        valid = 1-self._pb.get_terminal().flatten()
        o = self._obs
        a, agent_info = self._policy.action_np(o)
        next_o, r, d, env_info = self._env.step(a)
        t = self._pb.update(o,a,r,d,agent_info,env_info)
        self._obs = next_o
        self.step_id += 1
        if (np.all(t) and self.stop_if_terminal) or self.step_id >= self.max_path_length:
            self._handle_rollout_ending()
        return o,a,next_o,r,d,agent_info,env_info,valid

    def _start_new_rollout(self):
        self.step_id = 0
        self._obs = self._env.reset()
        self._pb = PathBuilder(len(self._obs))
        self._policy.reset()
        self.has_done = False

    def _handle_rollout_ending(self):
        if self.has_done:
            return
        temp_paths, temp_path_lens = self._pb.finalize(self._obs, aggregate=False, return_length=True)
        for i in range(len(temp_path_lens)):
            path_len = temp_path_lens[i]
            path = temp_paths[i]
            if self.cut:
                path = cut_path(path, path_len)
            if (
                path_len != self.max_path_length
                and not path['terminals'][path_len-1]
            ):
                if self.discard_incomplete_paths:
                    continue
                else:
                    assert self.cut
            self._num_paths_total += 1
            self._epoch_paths.append(path)
            self._epoch_path_lens.append(path_len)
        self.has_done = True
    
    def get_epoch_paths(self):
        return self._epoch_paths
    
    def start_epoch(self, epoch=None):
        self._epoch_num += 1
        self.wipe_memory()
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._epoch_path_lens = deque(maxlen=self._max_num_epoch_paths_saved)

    def end_epoch(self, epoch=None, start_new_epoch=False):
        self._handle_rollout_ending()
        if start_new_epoch:
            self.start_epoch()

    def get_diagnostics(self):
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length this epoch",
            list(self._epoch_path_lens),
            always_show_all_stats=True,
        ))
        return stats

class ExplorationStepCollector(SimpleStepCollector):
    def __init__(
        self,
        env,
        policy,
        qf,
        sample_number=32,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
    ):
        SimpleStepCollector.__init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved,
            render,
            render_kwargs,
        )
        self.sample_number = sample_number
        self._qf = qf

    def _collect_one_step(self):
        if self.has_done:
            self._start_new_rollout()
        valid = 1-self._pb.get_terminal().flatten()
        o = self._obs
        repeat_o = o.reshape(1,-1)
        repeat_o = np.tile(repeat_o, (self.sample_number,1))
        repeat_a, agent_info = self._policy.action_np(repeat_o)
        _, info = self._qf.value(ptu.from_numpy(repeat_o), ptu.from_numpy(repeat_a), only_return_ensemble=True)
        ensemble_atoms, _ = torch.min(info['ensemble_atoms'], dim=0)
        ensemble_atoms_std = torch.std(ensemble_atoms, dim=-1, keepdim=True)
        _, max_u_ind = torch.max(ensemble_atoms_std, dim=0)
        a = repeat_a[max_u_ind]
        a = a.reshape(1,-1)
        next_o, r, d, env_info = self._env.step(a)
        t = self._pb.update(o,a,r,d,agent_info,env_info)
        self._obs = next_o
        self.step_id += 1
        if (np.all(t) and self.stop_if_terminal) or self.step_id >= self.max_path_length:
            self._handle_rollout_ending()
        return o,a,next_o,r,d,agent_info,env_info,valid