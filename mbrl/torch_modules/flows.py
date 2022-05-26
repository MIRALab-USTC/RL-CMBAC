from mbrl.torch_modules.mlp import MLP
import torch
from torch import nn

# --------------------
# Model layers and helpers
# --------------------
class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self, x, return_log_det=True):
        sum_log_det = 0
        for module in self:
            x, log_det = module(x, return_log_det)
            if return_log_det:
                sum_log_det = sum_log_det + log_det
                
        if return_log_det:
            return x, sum_log_det
        else:
            return x, None

    def inverse(self, u, y):
        raise NotImplementedError


class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """
    def __init__(self, input_size, momentum=0.9, eps=1e-6):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:

            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0) 
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta
        log_det = self.log_gamma - 0.5 * torch.log(var + self.eps)
        return y, log_det.expand_as(x).sum(-1,keepdim=True)

class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """
    def __init__(self, input_size, mask, n_hidden=3):
        super().__init__()
        self.register_buffer('mask', mask)
        self.s_net = MLP(input_size,input_size,hidden_layers=[3]*n_hidden,nonlinearity='tanh',init_bias_constant=0)
        self.t_net = MLP(input_size,input_size,hidden_layers=[3]*n_hidden,nonlinearity='relu')

    def forward(self, x, return_log_det=True):
        # apply mask
        mx = x * self.mask
        # run through model
        s = self.s_net(mx)
        t = self.t_net(mx)
        h =  x * torch.exp(s) + t
        u = mx + (1-self.mask)*h

        if return_log_det:
            log_det = ((1-self.mask) * s).sum(-1, keepdim=True) 
        else:
            log_det = None 
        return u, log_det

    def inverse(self, u, return_abs_det=True):
        raise NotImplementedError

class RealNVP(nn.Module):
    def __init__(self, input_size, n_blocks=4, n_hidden=2, batch_norm=True):
        super().__init__()

        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, mask, n_hidden)]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(input_size)]
        self.net = FlowSequential(*modules)

    def forward(self, x, return_log_det=True):
        return self.net(x, return_log_det)

    def inverse(self, u, return_log_det=True):
        raise NotImplementedError


class Radial(nn.Module):
    def __init__(self, dim):
        super(Radial, self).__init__()
        self.z0 = nn.Parameter(torch.Tensor(1, dim))
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))
        self.dim = dim
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)

    def forward(self, z, return_log_det=True):
        r = torch.norm(z - self.z0, dim=-1).unsqueeze(-1)
        h = 1 / (self.alpha + r)
        out_put = z + (self.beta * h * (z - self.z0))
        
        if return_log_det:
            hp = - 1 / (self.alpha + r) ** 2
            bh = self.beta * h
            det_grad = (1 + bh) ** (self.dim - 1) * (1 + bh + self.beta * hp * r)
            log_abs_det = torch.log(det_grad.abs() + 1e-9)
        else:
            log_abs_det = None

        return out_put, log_abs_det

class RadialNormalizingFlow(nn.Module):
    def __init__(self, input_size, n_blocks=4):
        super().__init__()

        modules = []
        for i in range(n_blocks):
            modules += [Radial(input_size)]
        self.net = FlowSequential(*modules)

    def forward(self, x, return_log_det=True):
        return self.net(x, return_log_det)

    def inverse(self, u, return_log_det=True):
        raise NotImplementedError
