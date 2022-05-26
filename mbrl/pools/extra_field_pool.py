import numpy as np
import warnings
from mbrl.pools.simple_pool import SimplePool
import copy

class ExtraFieldPool(SimplePool):
    def __init__(self, env, max_size=1e6, extra_fields={}, compute_mean_std=False):
        super(ExtraFieldPool, self).__init__(env, max_size, compute_mean_std)
        self.extra_fields = {}
        self.required_samples = {}
        self.extra_fields_stop = {}
        self.extra_fields_size = {}
        self.add_extra_fields(extra_fields)
    
    @property
    def extra_keys(self):
        return list(self.extra_fields.keys())

    def add_extra_fields(self, extra_fields):
        for k,v in extra_fields.items():
            assert k not in self.fields
            if k in self.extra_fields:
                warnings.warn('Add a same extra_field [%s]. It will cover the old one.'%k)
            self.required_samples[k] = 0
            self.extra_fields_stop[k] = 0
            self.extra_fields_size[k] = 0
            self.initialize_field(k, v) 
        self.extra_fields.update(extra_fields)

    def add_samples(self, samples):
        new_sample_size = super(ExtraFieldPool, self).add_samples(samples)
        for key in self.extra_fields:
            self.required_samples[key] += new_sample_size
    
    def _check_keys(self, keys, without_keys):
        if keys is None:
            keys = list(self.fields.keys()) + list(self.extra_fields.keys())
            keys = [k for k in keys if k not in without_keys]
        for k in keys:
            if k in self.extra_fields:
                if self.required_samples[k] > 0:
                    raise RuntimeError("the %s data is not aligned with the sampled data."%(k))
            else:
                assert k in self.fields
        return keys
        
    def update_single_extra_field(self, key, value):
        assert key in self.extra_fields
        if self.compute_mean_std:
            self.dataset_mean_std[key].update(value)
        new_sample_size = len(value)
        max_size = self.max_size

        assert new_sample_size <= self.required_samples[key], "%d, %d"%(new_sample_size, self.required_samples[key])
        self.required_samples[key] -= new_sample_size
        stop = self.extra_fields_stop[key]

        self.extra_fields_stop[key] = new_stop = (stop + new_sample_size) % max_size
        cur_size = self.extra_fields_size[key]
        self.extra_fields_size[key] = min(max_size, cur_size + new_sample_size)
        
        if stop + new_sample_size >= max_size:
            self.dataset[key][stop:max_size] = value[:max_size-stop]
            self.dataset[key][:new_stop] = value[new_sample_size-new_stop:]
        else:
            self.dataset[key][stop:new_stop] = value

    def update_extra_fields(self, data):
        for k,v in data.items():
            assert k in self.extra_fields
            self.update_single_extra_field(k, v)
    
    def resize(self, size):
        extra_samples = self.get_data(self.extra_keys)
        extra_fields  = copy.deepcopy(self.extra_fields)
        super().resize(size,extra_fields=extra_fields)
        self.update_extra_fields(extra_samples)


            