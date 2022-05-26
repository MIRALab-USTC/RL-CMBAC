from mbrl.policies.base_policy import Policy
import numpy as np
from torch import nn

def make_determinsitic(random_policy):
    return MakeDeterministic(random_policy)

class MakeDeterministic(nn.Module, Policy):
    def __init__(self, random_policy):
        nn.Module.__init__(self)
        self.random_policy = random_policy

    def action(self, obs, return_info, **kwargs):
        with self.random_policy.set_deterministic(True):
            return self.random_policy.action(obs, return_info=return_info, **kwargs)

    def reset(self, **kwarg):
        self.random_policy.reset(**kwarg)

    def save(self, **kwarg):
        self.random_policy.save(**kwarg)

    def load(self, **kwarg):
        self.random_policy.load(**kwarg)
