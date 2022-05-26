import torch 
import numpy as np
import mbrl.torch_modules.utils as ptu

def reward_function(obs, action, next_obs, input_type='torch'):
    raise NotImplementedError

def done_function(obs, action, next_obs, input_type='torch'):
    height, ang = next_obs[...,0], next_obs[...,1]
    if input_type == "torch":
        finite = torch.isfinite(next_obs).all(-1)
        bounded = ( torch.abs(next_obs[...,1:])<100 ).all(-1) 
        abs_ang = torch.abs(ang)
    elif input_type == "numpy":
        finite = np.isfinite(next_obs).all(-1)
        bounded = ( np.abs(next_obs[...,1:])<100 ).all(-1) 
        abs_ang = np.abs(ang)
    else:
        raise NotImplementedError

    live = finite & bounded & (height > 0.7) & (abs_ang < 0.2)
    d = (~live)[...,None]

    if input_type == "torch":
        d = d.float()
    elif(input_type) == "numpy":
        d = d.astype(float)

    return d