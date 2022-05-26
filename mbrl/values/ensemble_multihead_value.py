import torch
from torch import nn
import numpy as np
from mbrl.torch_modules.mlp import MLP
import mbrl.torch_modules.utils as ptu
from mbrl.utils.logger import logger
from mbrl.values.base_value import StateValue, QValue

class EnsembleMultiheadQValue(nn.Module, QValue):
    def __init__( 
            self, 
            env, 
            ensemble_size=5,
            number_head=25, 
            with_target_value=True,
            value_name='ensemble_multihead_q_value',
            **mlp_kwargs
        ):
        nn.Module.__init__(self)
        QValue.__init__(self, env)
        self.with_target_value = with_target_value
        self.ensemble_size = ensemble_size
        self.number_head = number_head
        assert len(self.observation_shape) == 1 and len(self.action_shape) == 1
        self.module = MLP( self.observation_shape[0] + self.action_shape[0], 
                           number_head,
                           ensemble_size=ensemble_size,
                           module_name=value_name,
                           **mlp_kwargs)
        if with_target_value:
            self.target_module = MLP( 
                self.observation_shape[0] + self.action_shape[0], 
                number_head,
                ensemble_size=ensemble_size,
                module_name=value_name,
                **mlp_kwargs
            )
            self.update_target(1)

    def _sample_atoms(self, atoms, n_sample, replace):
        batch_shape = atoms.shape[:-1]
        n_atom = atoms.shape[-1]
        if replace:
            shape = batch_shape + (n_sample,)
            indices = ptu.randint(n_atom, shape)
        else:
            shape = batch_shape + (1,)
            for i in range(n_atom-n_sample, n_atom):
                n_choice = i + 1
                if i == n_atom-n_sample:
                    indices = ptu.randint(n_choice, shape)
                else:
                    temp_indices = ptu.randint(n_choice, shape)
                    indices = (indices + temp_indices + 1) % n_choice
                    indices = torch.cat([indices, temp_indices], -1)
        return torch.gather(atoms,-1,indices)

    def _mean_std_atoms(
        self,
        value_atoms, 
        alpha
    ):
        value_mean = torch.mean(value_atoms,dim=-1,keepdim=True)
        value_std = torch.std(value_atoms,dim=-1,unbiased=False,keepdim=True)
        value = value_mean - alpha * value_std
        return value, value_atoms

    def process_value_atoms(
        self,
        value_atoms, 
        redq_sample_mode
    ):
        if redq_sample_mode is None:
            value = torch.mean(value_atoms,dim=-1,keepdim=True)
        else:
            if redq_sample_mode == "mean":
                value = torch.mean(value_atoms,dim=-1,keepdim=True)
            elif redq_sample_mode == "min":
                value, _ = torch.min(value_atoms,dim=-1,keepdim=True)
            elif redq_sample_mode == "max":
                value, _ = torch.max(value_atoms,dim=-1,keepdim=True)
            else:
                raise NotImplementedError
        
        return value

    def _drop_atoms(
        self, 
        value_atoms, 
        number_selected,
        number_drop,
        percentage_drop,
        redq_sample_mode=None
    ):
        n_atom = value_atoms.shape[-1]
        if number_selected is None:
            if number_drop is None:
                if percentage_drop is None:
                    number_drop = 0
                else:
                    number_drop = round(n_atom*percentage_drop/100)
            number_selected = n_atom - number_drop
        else:
            number_drop = n_atom - number_selected

        if number_selected == 1:
            value,_ = torch.min(value_atoms,dim=-1,keepdim=True)
        elif number_drop > 0:
            value_atoms,_ = torch.sort(value_atoms, dim=-1) 
            value_atoms = value_atoms[...,:-number_drop]
            value = self.process_value_atoms(value_atoms, redq_sample_mode)
        elif number_drop == 0:
            value = self.process_value_atoms(value_atoms, redq_sample_mode)
        else:
            raise NotImplementedError
        return value, value_atoms

    def _redq_sample(
        self, 
        filtered_ensemble_atoms,
        redq_sample_num,
    ):
        n_atoms = filtered_ensemble_atoms.shape[-1]
        sample_indexes = np.arange(n_atoms)
        np.random.shuffle(sample_indexes)
        sample_indexes = sample_indexes[:redq_sample_num]
        sampled_ensemble_atoms = filtered_ensemble_atoms[...,sample_indexes]

        return sampled_ensemble_atoms

    def _get_q_uncertainty(
        self,
        value_atoms,
        number_drop,
        ensemble
    ):
        if number_drop is not None:
            value_atoms,_ = torch.sort(value_atoms, dim=-1) 
            value_atoms = value_atoms[...,:-number_drop]
        u = torch.std(value_atoms, dim=-1, keepdim=True)
        if ensemble:
            u = torch.mean(u, dim=0)
        
        return u

    def value(
        self, 
        obs, 
        action, 
        return_info=True, 
        use_target_value=False,

        atom_fliter=None, 
        number_sampled_atoms=None,
        sample_replace=False,

        number_selected=None,
        number_drop=None, 
        percentage_drop=None,

        mode="min_mean",

        redq_sample_num=None,
        redq_sample_mode=None,

        return_atoms=False,
        return_ensemble=False,
        only_return_ensemble=False,
        alpha=0.5,
        return_uncertainty=False,
    ):
        # drop_atoms: filter -> sample -> drop(sort, min)
        # xxx_mean: filter -> drop(sort, min) -> mean -> xxx
        # xxx = [min/max/mean/sample]
        if obs.dim() > 2:
            obs = obs.unsqueeze(-3)
            action = action.unsqueeze(-3)
        input_tensor = torch.cat([obs, action], dim=-1)
        ensemble = self.ensemble_size is not None

        # get ensemble atoms
        if ensemble:
            if use_target_value:
                ensemble_atoms = self.target_module(input_tensor)
            else:
                ensemble_atoms = self.module(input_tensor)

            #TODO network based sample
            #TODO ensemble value also need these
            if only_return_ensemble:
                info = {'ensemble_atoms': ensemble_atoms}
                return None, info

            if atom_fliter is not None:
                filtered_ensemble_atoms = ensemble_atoms[...,atom_fliter]
            else:
                filtered_ensemble_atoms = ensemble_atoms

            # redq style sample
            if redq_sample_num is not None:
                filtered_ensemble_atoms = self._redq_sample(
                    filtered_ensemble_atoms,
                    redq_sample_num
                )

        else:
            if use_target_value:
                value_atoms = self.target_module(input_tensor)
            else:
                value_atoms = self.module(input_tensor)

            if only_return_ensemble:
                info = {'ensemble_atoms': value_atoms}
                return None, info

            if atom_fliter is not None:
                filtered_value_atoms = value_atoms[...,atom_fliter]
            else:
                filtered_value_atoms = value_atoms

        # process ensemble atoms to get values
        if mode == "drop_atoms":
            if ensemble:
                filtered_value_atoms = filtered_ensemble_atoms.transpose(-3,-2)
                shape = filtered_value_atoms.shape[:-2] + (-1,)
                filtered_value_atoms = filtered_value_atoms.reshape(shape)

                if number_sampled_atoms is not None and \
                    number_sampled_atoms<shape[-1]:
                    sampled_value_atoms = self._sample_atoms(
                        filtered_value_atoms, 
                        number_sampled_atoms, 
                        sample_replace
                    )
                else:
                    sampled_value_atoms = filtered_value_atoms

            value, final_value_atoms = self._drop_atoms(
                sampled_value_atoms,
                number_selected,
                number_drop,
                percentage_drop
            )

        if mode[-5:] == "_mean":
            sub_mode = mode[:-5]
            if ensemble:
                if return_uncertainty:
                    u = self._get_q_uncertainty(filtered_ensemble_atoms, number_drop, ensemble)

                if sub_mode[:4] == "std_":
                    sub_mode = sub_mode[4:]
                    ensemble_value, final_value_atoms = self._mean_std_atoms(
                        filtered_ensemble_atoms,
                        alpha
                    )
                else:
                    ensemble_value, final_value_atoms = self._drop_atoms(
                        filtered_ensemble_atoms,
                        number_selected,
                        number_drop,
                        percentage_drop,
                        redq_sample_mode
                    )
                if sub_mode == 'min':
                    value = torch.min(ensemble_value, dim=-3)[0]
                elif sub_mode == 'mean':
                    value = torch.mean(ensemble_value, dim=-3)[0]
                elif sub_mode == 'max':
                    value = torch.max(ensemble_value, dim=-3)[0]
                elif sub_mode == 'sample':
                    index = np.random.randint(self.ensemble_size)
                    value = ensemble_value[index]
                else:
                    raise NotImplementedError
            else:
                if return_uncertainty:
                    u = self._get_q_uncertainty(filtered_value_atoms, number_drop, ensemble)
                if sub_mode[:4] == "std_":
                    sub_mode = sub_mode[4:]
                    value, final_value_atoms = self._mean_std_atoms(
                        filtered_value_atoms,
                        alpha
                    )
                else:
                    value, final_value_atoms = self._drop_atoms(
                        filtered_value_atoms,
                        number_selected,
                        number_drop,
                        percentage_drop,
                        redq_sample_mode
                    )
                    if obs.dim() > 2:
                        assert value.shape[-3] == 1
                        value = torch.min(value, dim=-3)[0]
        if return_info:
            info = {}
            if return_atoms:
                # NOTE: return selected value_atoms 
                # NOTE: for xxx_mean, it is the selected ensemble_atoms
                info['value_atoms'] = final_value_atoms

            if return_ensemble:
                if ensemble:
                    info['ensemble_atoms'] = ensemble_atoms
                    if mode[-5:] == "_mean":
                        info['ensemble_value'] = ensemble_value
                # else:
                #     info['ensemble_atoms'] = final_value_atoms
            if return_uncertainty:
                info['q_uncertainty'] = u
            return value, info
        else:
            return value

    def update_target(self, tau):
        if tau == 1:
            ptu.copy_model_params_from_to(self.module, self.target_module)
        else:
            ptu.soft_update_from_to(self.module, self.target_module, tau)

    def get_snapshot(self):
        return self.module.get_snapshot()