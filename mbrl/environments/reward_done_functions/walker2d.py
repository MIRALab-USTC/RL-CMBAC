import torch 
import numpy
import mbrl.torch_modules.utils as ptu

def reward_function(obs, action, next_obs, input_type='torch'):
    raise NotImplementedError

def done_function(obs, action, next_obs, input_type='torch'):
    height = next_obs[..., 0:1]
    angle = next_obs[..., 1:2]
    live = (height > 0.8) & (height < 2.0) \
           & (angle > -1.0) & (angle < 1.0)
    d = (~live)

    if input_type == "torch":
        d = d.float()
    elif input_type == "numpy":
        d = d.astype(float)

    return d