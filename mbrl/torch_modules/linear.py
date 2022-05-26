import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform
import mbrl.torch_modules.utils as ptu
import math

class Linear(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 which_nonlinearity='relu',
                 with_bias=True,
                 init_weight_mode='uniform',
                 init_bias_constant=None,
                 dense=False, 
                 batch_normalize=False,
                 bn_kwargs={},
                 ):
        if dense:
            self.final_out_features = out_features + in_features
        else:
            self.final_out_features = out_features
        self.dense = dense

        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.with_bias = with_bias
        self.init_weight_mode = init_weight_mode
        self.init_bias_constant = init_bias_constant
        if which_nonlinearity == 'identity':
            self.which_nonlinearity = 'linear'
        else:
            self.which_nonlinearity = which_nonlinearity

        self._get_parameters()
        self.reset_parameters()

        self.bn = batch_normalize
        if self.bn:
            self.bn_layer = nn.BatchNorm1d(self.out_features, **bn_kwargs)
    
    def _get_parameters(self):
        self.weight, self.bias = self._creat_weight_and_bias()
    
    def _creat_weight_and_bias(self):
        weight = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        if self.with_bias:
            bias = nn.Parameter(torch.Tensor(1, self.out_features))
        else:
            bias = None
        return weight, bias

    def reset_parameters(self, gain_coef=1):
        self._reset_weight_and_bias(self.weight, self.bias, gain_coef)

    def _reset_weight_and_bias(self, weight, bias, gain_coef):
        ptu.fanin_init(weight, 
                       nonlinearity=self.which_nonlinearity, 
                       mode=self.init_weight_mode, 
                       gain_coef=gain_coef)
        if bias is not None:
            if self.init_bias_constant is None:
                fan_in = self.in_features
                bound = gain_coef / math.sqrt(fan_in)
                nn.init.uniform_(bias, -bound, bound)
            else:
                nn.init.constant_(bias, self.init_bias_constant)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.with_bias)

    def forward(self, x):
        while x.dim() < 2:
            x = x.unsqueeze(0)

        if self.with_bias:
            output = x.matmul(self.weight) + self.bias
        else:
            output = x.matmul(self.weight)

        if self.bn:
            assert output.dim() < 4
            output = self.bn_layer(output)
        
        if self.dense:
            output = torch.cat([output, x], -1)
        
        return output

    def get_weight_decay(self, weight_decay=5e-5):
        return (self.weight ** 2).sum() * weight_decay * 0.5
        

class EnsembleLinear(Linear):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 ensemble_size,
                 which_nonlinearity='relu',
                 with_bias=True,
                 init_weight_mode='uniform',
                 init_bias_constant=None,
                 dense = False,
                 batch_normalize=False,
                 bn_kwargs={},
                 ):
        if dense:
            self.final_out_features = out_features + in_features
        else:
            self.final_out_features = out_features
        self.dense = dense

        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.with_bias = with_bias
        self.init_weight_mode = init_weight_mode
        self.init_bias_constant = init_bias_constant
        if which_nonlinearity == 'identity':
            self.which_nonlinearity = 'linear'
        else:
            self.which_nonlinearity = which_nonlinearity

        self._get_parameters()
        self.reset_parameters()

        self.bn = batch_normalize
        self.bn_layers = []
        if self.bn:
            for i in range(self.ensemble_size):
                bn_name = "bn_net%d"%i
                bn_layer = nn.BatchNorm1d(self.out_features, **bn_kwargs)
                self.bn_layers.append(bn_layer)
                setattr(self, bn_name, bn_layer)
    
    def _get_parameters(self):
        self.weights, self.biases = [], []
        for i in range(self.ensemble_size):
            weight, bias = self._creat_weight_and_bias()
            weight_name, bias_name = 'weight_net%d'%i, 'bias_net%d'%i
            self.weights.append(weight)
            self.biases.append(bias)
            setattr(self, weight_name, weight)
            setattr(self, bias_name, bias)

    def reset_parameters(self, gain_coef=1):
        for w,s in zip(self.weights, self.biases):
            self._reset_weight_and_bias(w, s, gain_coef)

    def extra_repr(self):
        return 'in_features={}, out_features={}, ensemble_size={}, bias={}'.format(
            self.in_features, self.out_features, self.ensemble_size, self.with_bias)

    def forward(self, x):
        while x.dim() < 3:
            x = x.unsqueeze(0)

        w = torch.stack(self.weights, 0)
        if self.with_bias:
            b = torch.stack(self.biases, 0)
            output = x.matmul(w) + b
        else:
            output = x.matmul(w)

        if self.bn:
            assert output.dim() == 3
            norm_output = []
            for bn_layer, o in zip(self.bn_layers, output):
                norm_output.append(bn_layer(o))
            output = torch.stack(norm_output, 0)

        if self.dense:
            if x.dim() == 3 and x.shape[0] == 1:
                x = x.repeat((self.ensemble_size, 1, 1))
            output = torch.cat([output, x], -1)
        
        return output

    def get_weight_decay(self, weight_decay=5e-5):
        decays = []
        for w in self.weights:
            decays.append((w ** 2).sum() * weight_decay * 0.5)
        return sum(decays)


class NoisyLinear(Linear):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 noise_type='gaussian', #uniform 
                 factorised=False,
                 **linear_kwargs
                 ):
        assert noise_type in ['gaussian', 'uniform']
        self.noise_type = noise_type
        self.factorised = factorised
        super(NoisyLinear, self).__init__(in_features,
                                          out_features,
                                          **linear_kwargs
                                          )
        self.reset_parameters(gain_coef=0.5)
        self._get_sigma_parameters()
        self.reset_sigma_parameters()

    def _get_epsilon(self,
                     weight_epsilon_1_size,
                     weight_epsilon_2_size,
                     weight_epsilon_size,
                     bias_epsilon_size):
        if self.factorised:
            if self.noise_type == 'gaussian':
                weight_epsilon_1 = ptu.randn(weight_epsilon_1_size)
                weight_epsilon_2 = ptu.randn(weight_epsilon_2_size)
            elif self.noise_type == 'uniform':
                weight_epsilon_1 = ptu.rand(weight_epsilon_1_size)*2-1
                weight_epsilon_2 = ptu.rand(weight_epsilon_2_size)*2-1

            weight_epsilon = torch.matmul(weight_epsilon_1, weight_epsilon_2)
        else:
            if self.noise_type == 'gaussian':
                weight_epsilon = ptu.randn(weight_epsilon_size)
            elif self.noise_type == 'uniform':
                weight_epsilon = ptu.rand(weight_epsilon_size)*2-1

        if self.with_bias:
            if self.noise_type == 'gaussian':
                bias_epsilon = ptu.randn(bias_epsilon_size)
            elif self.noise_type == 'uniform': 
                bias_epsilon = ptu.rand(bias_epsilon_size)*2-1
        else:
            bias_epsilon = None

        return weight_epsilon, bias_epsilon
    
    def _get_sigma_parameters(self):
        self.sigma_weight, self.sigma_bias = self._creat_weight_and_bias()

    def reset_sigma_parameters(self, scale=1):
        self._reset_sigma_weight_and_bias(self.sigma_weight, self.sigma_bias, scale)
    
    def _reset_sigma_weight_and_bias(self, sigma_weight, sigma_bias, scale):
        fan_in = self.in_features
        nn.init.constant_(sigma_weight, scale / math.sqrt(fan_in))
        if sigma_bias is not None:
            nn.init.constant_(sigma_bias, scale / math.sqrt(fan_in))
    
    def forward(self, x, deterministic=False):
        if deterministic:
            mean = super(NoisyLinear, self).forward(x)
            return mean

        while x.dim() < 3:
            x = x.unsqueeze(0)

        batch_size = x.shape[-3]
        weight_epsilon_1_size = (batch_size, self.in_features, 1)
        weight_epsilon_2_size = (batch_size, 1, self.out_features)
        weight_epsilon_size = (batch_size, self.in_features, self.out_features)
        bias_epsilon_size = (batch_size, 1, self.out_features)
        weight_epsilon, bias_epsilon = self._get_epsilon(weight_epsilon_1_size,
                                                         weight_epsilon_2_size,
                                                         weight_epsilon_size,
                                                         bias_epsilon_size)

        weight_noise = weight_epsilon * self.sigma_weight
        weight_noise.requires_grad_(True)
        
        if self.with_bias:
            bias_noise = bias_epsilon * self.sigma_bias
            bias_noise.requires_grad_(True)
            return x.matmul(self.weight + weight_noise) + self.bias  + bias_noise
        else:
            return x.matmul(self.weight + weight_noise)
            


class NoisyEnsembleLinear(NoisyLinear, EnsembleLinear):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 ensemble_size,
                 noise_type='gaussian', #uniform 
                 factorised=False,
                 **ensemble_linear_kwargs
                 ):
        assert noise_type in ['gaussian', 'uniform']
        self.noise_type = noise_type
        self.factorised = factorised
        EnsembleLinear.__init__(self,  
                                in_features,
                                out_features,
                                ensemble_size,
                                **ensemble_linear_kwargs
                                )
        self.reset_parameters(gain_coef=0.5)
        self._get_sigma_parameters()
        self.reset_sigma_parameters()
    
    def _get_sigma_parameters(self):
        self.sigma_weights, self.sigma_biases = [], []
        for i in range(self.ensemble_size):
            sigma_weight, sigma_bias = self._creat_weight_and_bias()
            sigma_weight_name, sigma_bias_name = 'sigma_weight_net%d'%i, 'sigma_bias_net%d'%i
            self.sigma_weights.append(sigma_weight)
            self.sigma_biases.append(sigma_bias)
            setattr(self, sigma_weight_name, sigma_weight)
            setattr(self, sigma_bias_name, sigma_bias)

    def reset_sigma_parameters(self, scale=1):
        for w,s in zip(self.sigma_weights, self.sigma_biases):
            self._reset_sigma_weight_and_bias(w, s, scale)
    
    def forward(self, x, deterministic=False):
        if deterministic:
            mean = super(NoisyLinear, self).forward(x)
            return mean
        
        while x.dim() < 4:
            x = x.unsqueeze(0)

        batch_size = x.shape[-4]
        weight_epsilon_1_size = (batch_size, self.ensemble_size, self.in_features, 1)
        weight_epsilon_2_size = (batch_size, self.ensemble_size, 1, self.out_features)
        weight_epsilon_size = (batch_size, self.ensemble_size, self.in_features, self.out_features)
        bias_epsilon_size = (batch_size, self.ensemble_size, 1, self.out_features)
        weight_epsilon, bias_epsilon = self._get_epsilon(weight_epsilon_1_size,
                                                         weight_epsilon_2_size,
                                                         weight_epsilon_size,
                                                         bias_epsilon_size)

        sigma_weight = torch.stack(self.sigma_weights, 0)
        weight_noise = weight_epsilon * sigma_weight
        weight_noise.requires_grad_(True)

        w = torch.stack(self.weights, 0)
        if self.with_bias:
            b = torch.stack(self.biases, 0)
            sigma_bias = torch.stack(self.sigma_biases, 0)
            bias_noise = bias_epsilon * sigma_bias
            bias_noise.requires_grad_(True)
            return x.matmul(w + weight_noise) + b + bias_noise
        else:
            return x.matmul(w + weight_noise)
