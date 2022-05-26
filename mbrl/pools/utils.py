import numpy as np
from collections import OrderedDict

def get_unnested_data(data, pre_key=''):
    inner_dict = {}
    keys = list(data.keys())
    for key in keys:
        value = data.pop(key)
        if type(value) in [OrderedDict, dict]:
            inner_dict.update(get_unnested_data(value, pre_key=pre_key+key+'.'))
        else:
            data[pre_key+key] = value
    data.update(inner_dict)
    return data

def get_valid_dataset_size(dataset, keys=None):
    if keys is None:
        keys = list(dataset.keys())
    min_size = np.inf
    for key in keys:
        value = dataset[key]
        cur_size = len(value)
        if cur_size < min_size:
            min_size = cur_size
    return min_size

def get_batch(dataset, batch_index, keys=None):
    if keys is None:
        keys = list(dataset.keys())
    batch = {}
    for key in keys:
        value = dataset[key]
        batch[key] = value[batch_index]
    return batch

def _random_batch_independently(dataset, batch_size, valid_size, keys=None):
    batch_index = np.random.randint(0, valid_size, batch_size)
    return get_batch(dataset, batch_index, keys=keys)

def random_batch_independently(dataset, batch_size, keys=None):
    valid_size = get_valid_dataset_size(dataset, keys=keys)
    return _random_batch_independently(dataset, batch_size, valid_size, keys)

def _shuffer_and_random_batch(dataset, batch_size, valid_size, keys=None):
    _batch_index = np.random.permutation(np.arange(valid_size))
    ts = 0 
    while ts < valid_size:
        te = ts + batch_size
        if te + batch_size > valid_size:
            te += batch_size
        yield get_batch(dataset, _batch_index[ts:te], keys=keys)
        ts = te

def shuffer_and_random_batch(dataset, batch_size, keys=None):
    valid_size = get_valid_dataset_size(dataset, keys=keys)
    for batch in _shuffer_and_random_batch(dataset, batch_size, valid_size, keys):
        yield batch


def split_dataset(dataset, valid_ratio, max_valid = 5000, return_length=True):
    train_dataset = {}
    valid_dataset = {}
    n_sample = None
    for k, data in dataset.items():
        if n_sample is None:
            n_sample = len(data)
            valid_length = int( np.ceil(min(max_valid, valid_ratio*n_sample)) )
            train_length = n_sample - valid_length
            indices = np.random.permutation(n_sample)
            train_indices = indices[:train_length]
            valid_indices = indices[-valid_length:]
        else:
            assert n_sample == len(data)
        train_dataset[k] = data[train_indices]
        valid_dataset[k] = data[valid_indices]
    if return_length:
        return train_dataset, valid_dataset, train_length, valid_length
    else:
        return train_dataset, valid_dataset
        
