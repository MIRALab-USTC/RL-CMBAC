from mbrl.policies.base_policy import RandomPolicy
from mbrl.torch_modules.policies import MeanLogstdGaussianPolicyModule, TanhPolicyModule
from mbrl.utils.logger import logger
from torch import nn

class GaussianPolicy(nn.Module, RandomPolicy):
    def __init__( self, 
                  env, 
                  deterministic=False,
                  tanh_action=True,
                  policy_name='gaussian_policy',
                  **mlp_kwargs):
        nn.Module.__init__(self)
        RandomPolicy.__init__(self, env, deterministic)
        assert len(self.observation_shape) == 1 and len(self.action_shape) == 1
        gaussian = MeanLogstdGaussianPolicyModule( self.observation_shape[0], 
                                                   self.action_shape[0],
                                                   policy_name,
                                                   **mlp_kwargs)
        self.tanh_action = tanh_action

        if tanh_action:
            self.module = TanhPolicyModule(gaussian)
        else:
            self.module = gaussian
        
    def action( self, 
                 obs, 
                 return_info=True,
                 return_log_prob=False,
                 **kwargs
                ):
        return self.module(obs, 
                            deterministic=self._deterministic, 
                            return_info=return_info, 
                            return_log_prob=return_log_prob,
                            **kwargs)

    def log_prob(self, obs, action):
        return self.module.log_prob(obs, action)

    def save(self, save_dir=None):
        if save_dir == None:
            save_dir = logger._snapshot_dir
        self.module.save(save_dir)
    
    def load(self, load_dir=None):
        if load_dir == None:
            load_dir = logger._snapshot_dir
        self.module.load(load_dir)

    def get_snapshot(self):
        return self.module.get_snapshot()


