from mbrl.processors.base_processor import Processor
import torch
import torch.nn as nn
import mbrl.torch_modules.utils as ptu

class Normalizer(nn.Module, Processor):
    def __init__(self, shape, epsilon=1e-6):
        nn.Module.__init__(self)
        self.input_shape = self.output_shape = shape
        self.epsilon = ptu.FloatTensor([epsilon])
        self.mean = nn.Parameter(ptu.zeros(shape), requires_grad=False)
        self.std = nn.Parameter(ptu.ones(shape), requires_grad=False)
    
    def forward(self, x):
        return self.process(x)

    def process(self, x):    
        return (x-self.mean) / (self.std + self.epsilon)

    def recover(self, x):
        return x * (self.std + self.epsilon) + self.mean

    def set_mean_std_np(self, mean, std):
        self.mean.data = ptu.from_numpy(mean)
        self.std.data = ptu.from_numpy(std)

    def mean_std_np(self):
        return ptu.get_numpy(self.mean.data), ptu.get_numpy(self.std.data)