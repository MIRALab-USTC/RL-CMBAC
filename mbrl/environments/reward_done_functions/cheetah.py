import torch 
import numpy
import mbrl.torch_modules.utils as ptu

def reward_function(obs, action, next_obs, input_type='torch'):
    raise NotImplementedError

def done_function(obs, action, next_obs, input_type='torch'):
    size = next_obs.size()
    if input_type == 'torch':
        return ptu.zeros(size)[...,0:1]
    elif input_type == 'numpy':
        return numpy.zeros(size)[...,0:1]
    else:
        raise NotImplementedError