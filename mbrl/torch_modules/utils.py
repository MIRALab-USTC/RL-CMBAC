import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import math

"""
GPU wrappers
"""
_use_gpu = False
device = None
_gpu_id = 0

def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")
    if _use_gpu:
        set_device(gpu_id)

def gpu_enabled():
    return _use_gpu

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

# noinspection PyPep8Naming
def FloatTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.FloatTensor(*args, **kwargs).to(torch_device)

def LongTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.LongTensor(*args, **kwargs, device=torch_device)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)

def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)

def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)

def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)

def randn_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn_like(*args, **kwargs, device=torch_device)

def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)

def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)

def rand(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.rand(*args, **kwargs, device=torch_device)

def rand_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.rand_like(*args, **kwargs, device=torch_device)

def randint(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randint(*args, **kwargs, device=torch_device)

def arange(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.arange(*args, **kwargs, device=torch_device)


############################ our utils ############################

class Swish(nn.Module):
    def __init__(self):        
        super(Swish, self).__init__()     
    def forward(self, x):        
        x = x * torch.sigmoid(x)        
        return x

class Identity(nn.Module):
    def __init__(self):        
        super(Identity, self).__init__()     
    def forward(self, x):            
        return x

swish = Swish()
identity = Identity()

def get_nonlinearity(act_name='relu'):
    nonlinearity_dict = {
        'relu': F.relu,
        'swish': swish,
        'tanh': torch.tanh,
        'identity': identity,
    }
    return nonlinearity_dict[act_name]

def fanin_init(tensor, nonlinearity='relu', mode='uniform', gain_coef=None):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    else:
        raise Exception("Shape must be have dimension 2.")
    if gain_coef is None:
        gain_coef = 1
        
    try:
        gain = gain_coef * init.calculate_gain(nonlinearity)
    except:
        gain = gain_coef * init.calculate_gain('relu')

    if mode == 'uniform':
        bound = gain * math.sqrt(3.0) / np.sqrt(fan_in)
        return tensor.data.uniform_(-bound, bound)
    elif mode == 'normal':
        std = gain / np.sqrt(fan_in)
        return tensor.data.normal_(-std, std)

def fanin_init_weights_like(tensor):
    new_tensor = FloatTensor(tensor.size())
    fanin_init(new_tensor)
    return new_tensor

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return from_numpy(elem_or_tuple).float()

def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v

def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for (k, x) in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }

def _elem_or_tuple_to_numpy(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return get_numpy(elem_or_tuple)

def torch_to_np_info(torch_info):
    return {
        k: _elem_or_tuple_to_numpy(x)
        for (k, x) in torch_info.items()
    }
