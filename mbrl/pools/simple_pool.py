import numpy as np
import warnings
from collections import OrderedDict

from mbrl.utils.mean_std import RunningMeanStd
from mbrl.pools.base_pool import Pool
from mbrl.pools.utils import get_batch, _random_batch_independently, _shuffer_and_random_batch
from mbrl.collectors.utils import path_to_samples
from mbrl.utils.eval_util import create_stats_ordered_dict
from mbrl.global_config import *
import copy

class SimplePool(Pool):
    def __init__(self, env, max_size=1e6, compute_mean_std=False):
        self._env = env
        self.max_size = int(max_size)
        self.compute_mean_std = compute_mean_std
        o_shape = self._observation_shape = self._env.observation_space.shape
        a_shape = self._action_shape = self._env.action_space.shape
        self.fields = {
            'observations': {
                'shape': o_shape,
                'type': np.float,
            },
            'next_observations': {
                'shape': o_shape,
                'type': np.float,
            },
            'actions': {
                'shape': a_shape,
                'type': np.float,
            },
            'rewards': {
                'shape': (1,),
                'type': np.float,
            },
            'terminals': {
                'shape': (1,),
                'type': np.float,
            },
        }
        self.sample_keys = list(self.fields.keys())
        self.dataset = {}
        if self.compute_mean_std:
            self.dataset_mean_std = {}
        for k, v in self.fields.items():
            self.initialize_field(k, v) 
        self._size = 0
        self._stop = 0
        self.unprocessed_size={}

    def initialize_field(self, field_name, field_info):
        self.dataset[field_name] = np.empty((int(self.max_size), *field_info['shape']), dtype=field_info['type'])
        if self.compute_mean_std:
            self.dataset_mean_std[field_name] = RunningMeanStd(field_info['shape'])

    def get_size(self):
        return self._size

    def get_mean_std(self, keys=None, without_keys=[]):
        assert self.compute_mean_std
        keys = self._check_keys(keys, without_keys)
        mean_std = {
            k: [self.dataset_mean_std[k].mean,
                self.dataset_mean_std[k].std]
            for k in keys
        }
        return mean_std
    
    def random_batch(self, batch_size, keys=None, without_keys=[]):
        keys = self._check_keys(keys, without_keys)
        return _random_batch_independently(self.dataset, batch_size, self._size, keys)

    def shuffer_and_random_batch(self, batch_size, keys=None, without_keys=[]):
        keys = self._check_keys(keys, without_keys)
        for batch in _shuffer_and_random_batch(self.dataset, batch_size, self._size, keys):
            yield batch

    def get_data(self, keys=None, without_keys=[]):
        keys = self._check_keys(keys, without_keys)
        data = {}
        for k in keys:
            temp_data = self.dataset[k]
            if self._size < self.max_size:
                data[k] = temp_data[:self._size]
            else:
                stop, size = self._stop, self._size
                data[k] = np.concatenate((temp_data[stop-size:], temp_data[:stop]))
        return data

    def _update_single_field(self, key, value):
        assert key in self.fields
        if self.compute_mean_std:
            self.dataset_mean_std[key].update(value)
        new_sample_size = len(value)
        max_size = self.max_size
        stop = self._stop
        new_stop = (stop + new_sample_size) % max_size
        if stop + new_sample_size >= max_size:
            self.dataset[key][stop:max_size] = value[:max_size-stop]
            self.dataset[key][:new_stop] = value[new_sample_size-new_stop:]
        else:
            self.dataset[key][stop:new_stop] = value

    def add_paths(self, paths):
        self.add_samples(path_to_samples(paths))
    
    def add_samples(self, samples):
        for k in self.fields:
            v = samples[k]
            self._update_single_field(k,v)
        stop = self._stop
        new_sample_size = len(samples[k])
        max_size = self.max_size
        self._stop = new_stop = (stop + new_sample_size) % max_size
        self._size = min(max_size, self._size + new_sample_size)
        for tag in self.unprocessed_size:
            unprocessed_size = self.unprocessed_size[tag] + new_sample_size
            if unprocessed_size > max_size:
                warnings.warn("unprocessed_size > max_size")
                self.unprocessed_size[tag] = max_size
            else:
                self.unprocessed_size[tag] = unprocessed_size
        return new_sample_size
    
    def get_unprocessed_data(self, tag, keys=None, without_keys=[]):
        keys = self._check_keys(keys, without_keys)
        if tag not in self.unprocessed_size:
            self.unprocessed_size[tag] = self._size
        stop = self._stop
        size = self.unprocessed_size[tag]
        data = {}
        for key in keys:
            assert key in self.fields
            temp_data = self.dataset[key]
            if size > stop:
                data[key] = np.concatenate((temp_data[stop-size:], temp_data[:stop]))
            else:
                data[key] = temp_data[stop-size:stop]
        return data

    def update_process_flag(self, tag, process_num):
        if tag not in self.unprocessed_size:
            self.unprocessed_size[tag] = self._size
        assert process_num <= self.unprocessed_size[tag]
        self.unprocessed_size[tag] -= process_num
    
    def resize(self, size, **init_kwargs):
        assert size > self._size
        samples = self.get_data(self.sample_keys)
        unprocessed_size = copy.deepcopy(self.unprocessed_size)
        self.__init__(self._env, size, compute_mean_std=self.compute_mean_std, **init_kwargs)
        self.add_samples(samples)
        self.unprocessed_size = unprocessed_size

    def _check_keys(self, keys, without_keys):
        if keys is None:
            keys = [k for k in self.fields.keys() if k not in without_keys]
        for k in keys:
            assert k in self.fields
        return keys
        
    def get_diagnostics(self):
        diagnostics =  OrderedDict([
            ('size', self._size),

        ])
        if LOG_LEVEL >= FULL_INFO:
            diagnostics.update(create_stats_ordered_dict(
                'Reward',
                self.dataset['rewards'],
            ))
            d = self.dataset['terminals']
            diagnostics['Done Ratio'] = d.sum() / self._size
        return diagnostics

        