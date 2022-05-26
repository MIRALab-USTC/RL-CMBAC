import torch
from torch import nn
import numpy as np
from mbrl.torch_modules.mlp import MLP
import mbrl.torch_modules.utils as ptu
from mbrl.utils.logger import logger
from mbrl.values.base_value import StateValue, QValue
from mbrl.models.pe_model import sample_from_ensemble
class EnsembleQValue(nn.Module, QValue):
    def __init__( 
        self, 
        env, 
        ensemble_size=2,
        with_target_value=True,
        value_name='ensemble_q_value',
        **mlp_kwargs
    ):
        nn.Module.__init__(self)
        QValue.__init__(self, env)
        self.with_target_value = with_target_value
        self.ensemble_size = ensemble_size
        assert len(self.observation_shape) == 1 and len(self.action_shape) == 1
        self.module = MLP( self._get_feature_size(), 
                           1,
                           ensemble_size=ensemble_size,
                           module_name=value_name,
                           **mlp_kwargs)
        if with_target_value:
            self.target_module = MLP( self._get_feature_size(), 
                               1,
                               ensemble_size=ensemble_size,
                               module_name=value_name,
                               **mlp_kwargs)
            self.update_target(1)
    
    def _get_feature_size(self):
        return self.observation_shape[0] + self.action_shape[0]

    def _get_features(self, obs, action):
        if obs.dim() > 2:
            obs = obs.unsqueeze(-3)
            action = action.unsqueeze(-3)
        return torch.cat([obs, action], dim=-1)

    def value(
        self, 
        obs, 
        action, 
        return_info=True, 
        use_target_value=False,
        sample_number=2, 
        batchwise_sample=False,
        mode='min', 
        return_ensemble=False
    ):
        input_tensor = self._get_features(obs, action)

        if self.ensemble_size is not None:
            if use_target_value:
                ensemble_value = self.target_module(input_tensor)
            else:
                ensemble_value = self.module(input_tensor)

            if self.ensemble_size != sample_number:
                if batchwise_sample:
                    index = np.random.choice(self.ensemble_size, sample_number, replace=False)
                    ensemble_value = ensemble_value[...,index,:,:]
                else:
                    ensemble_value = sample_from_ensemble(ensemble_value, sample_number, replace=False)

            if mode == 'min':
                value = torch.min(ensemble_value, dim=-3)[0]
            elif mode == 'mean':
                value = torch.mean(ensemble_value, dim=-3)[0]
            elif mode == 'max':
                value = torch.max(ensemble_value, dim=-3)[0]
            elif mode == 'sample':
                index = np.random.randint(self.ensemble_size)
                value = ensemble_value[index]
            else:
                raise NotImplementedError

        else:
            if use_target_value:
                value = self.target_module(input_tensor)
            else:
                value = self.module(input_tensor)
            if obs.dim() > 2:
                assert value.shape[-3] == 1
                value = torch.min(value, dim=-3)[0]
        if return_info:
            info = {}
            if return_ensemble:
                if self.ensemble_size is not None:
                    info['ensemble_value'] = ensemble_value
                else:
                    info['ensemble_value'] = value
            return value, info
        else:
            return value

    def update_target(self, tau):
        if tau == 1:
            ptu.copy_model_params_from_to(self.module, self.target_module)
        else:
            ptu.soft_update_from_to(self.module, self.target_module, tau)

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

class LatentEnsembleQValue(EnsembleQValue):
    def __init__( self, 
                  env, 
                  pair_processor,
                  ensemble_size=2, 
                  with_target_value=True,
                  value_name='ensemble_q_value',
                  **mlp_kwargs
                ):
        self.feature_size = pair_processor.output_shape[0]
        super().__init__(env, ensemble_size, with_target_value,value_name,**mlp_kwargs)
        self.pair_processor = pair_processor
    
    def _get_feature_size(self):
        return self.feature_size
    
    def _get_features(self, obs, action):
        output = self.pair_processor.process(obs, action)
        return output