import torch 
import numpy
import mbrl.torch_modules.utils as ptu

def reward_function(obs, action, next_obs, input_type='torch'):
    raise NotImplementedError

def done_function(obs, action, next_obs, input_type='torch'):
    qpos2 = next_obs[...,0:1]
    d = (qpos2 < 1.0) | (qpos2 > 2.0)
    
    if input_type == "torch":
        d = d.float()
    elif(input_type) == "numpy":
        d = d.astype(float)
    
    return d