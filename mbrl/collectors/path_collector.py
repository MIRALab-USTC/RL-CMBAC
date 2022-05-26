from collections import deque, OrderedDict
import numpy as np

from mbrl.utils.eval_util import create_stats_ordered_dict
from mbrl.collectors.utils import rollout, cut_path
from mbrl.collectors.base_collector import PathCollector
from contextlib import contextmanager


class SimplePathCollector(PathCollector):
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
    
    def set_policy(self, policy, epoch=None):
        self.end_epoch(epoch)
        self._policy = policy
    
    @contextmanager
    def with_policy(self, policy):
        old_policy = self._policy
        self.set_policy(policy)
        yield
        self.set_policy(old_policy)


    def collect_new_paths(
            self,
            num_steps,
            max_path_length=1000,
            discard_incomplete_paths=True,
            cut=True,
            stop_if_terminal=True,
    ):
        max_path_length = min(self._env.horizon,max_path_length)
        paths = []
        path_lens = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            temp_paths, temp_path_lens = rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length,
                aggregate=False,
                return_length=True,
                stop_if_terminal=stop_if_terminal,
            )
            i = 0
            while i < len(temp_path_lens) and num_steps_collected < num_steps:
                path_len = temp_path_lens[i]
                if num_steps_collected + path_len > num_steps:
                    path_len = num_steps-num_steps_collected
                path = temp_paths[i]
                if cut:
                    path = cut_path(path, path_len)
                if (
                        path_len != max_path_length
                        and not path['terminals'][path_len-1]
                ):
                    if discard_incomplete_paths:
                        break
                    else:
                        assert cut
                num_steps_collected += path_len
                paths.append(path)
                path_lens.append(path_len)
                i = i + 1
            if i < len(temp_path_lens):
                break
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        self._epoch_path_lens.extend(path_lens)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def start_epoch(self, epoch=None):
        self._epoch_num += 1
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._epoch_path_lens = deque(maxlen=self._max_num_epoch_paths_saved)

    def end_epoch(self, epoch=None, start_new_epoch=False):
        if start_new_epoch:
            self.start_epoch()

    def get_diagnostics(self):
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            list(self._epoch_path_lens),
            always_show_all_stats=True,
        ))
        return stats
