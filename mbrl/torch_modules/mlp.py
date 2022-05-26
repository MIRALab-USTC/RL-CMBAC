import torch
import os.path as osp
import torch.nn as nn

import mbrl.torch_modules.utils as ptu
from mbrl.torch_modules.linear import *
from mbrl.utils.misc_untils import to_list
import copy

class MLP(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_layers=[128,128], 
                 ensemble_size=None,
                 nonlinearity='relu', 
                 output_nonlinearity='identity',
                 module_name='mlp',
                 **fc_kwargs
                 ):
        #If ensemble is n
        #Given a tensor with shape (n,a,b) output (n,a,c)
        #Given a tensor with shape (a,b) output (n,a,c).  

        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layers = len(hidden_layers)
        self.ensemble_size = ensemble_size
        self.module_name = module_name

        #get activation functions
        self.output_nonlinearity = output_nonlinearity
        self.nonlinearities = to_list(nonlinearity, self.num_hidden_layers)
        self.nonlinearities.append(output_nonlinearity)
        self.activation_functions = [ptu.get_nonlinearity(nl) for nl in self.nonlinearities]

        self.layers = [input_size] + hidden_layers + [output_size]
        self.fc_kwargs = fc_kwargs
        self._get_layers()
        self.min_output_dim = 2 if self.ensemble_size is None else 3

    def _get_layers(self):
        self.fcs = []
        self.final_layers = [self.layers[0]]  # for densenet

        in_features = self.layers[0]
        for i in range(len(self.layers)-1):
            fc = self._get_single_layer(i, in_features)
            in_features = fc.final_out_features

            self.final_layers.append(in_features) # for densenet
            
            setattr(self, 'layer%d'%i, fc)
            self.fcs.append(fc)

    def _get_single_layer(self, i, in_features):
        fc_kwargs = copy.deepcopy(self.fc_kwargs)

        if i == self.num_hidden_layers: # last layer should not be dense
            fc_kwargs['dense'] = False        

        if self.ensemble_size is None:
            fc = Linear(in_features,
                        self.layers[i+1],
                        which_nonlinearity=self.nonlinearities[i],
                        **fc_kwargs)
        else:
            fc = EnsembleLinear(in_features, 
                                self.layers[i+1], 
                                self.ensemble_size,
                                which_nonlinearity=self.nonlinearities[i],
                                **fc_kwargs)
        return fc
    
    def get_snapshot(self, key_must_have=''):
        new_state_dict = {}
        state_dict = self.state_dict()
        if key_must_have == '':
            new_state_dict = state_dict
        else:
            for k,v in state_dict.items():
                if key_must_have in k:
                    new_state_dict[k] = v
        return new_state_dict

    def load_snapshot(self, loaded_state_dict, key_must_have=''):
        state_dict = self.state_dict()
        if key_must_have == '':
            state_dict = loaded_state_dict
        else:
            for k,v in loaded_state_dict.items():
                if key_must_have in k:
                    state_dict[k] = v
        self.load_state_dict(state_dict)

    def save(self, save_dir, net_id=None):
        if self.ensemble_size is None or net_id is None:
            net_name = ''
            file_path = osp.join(save_dir, '%s.pt'%self.module_name)
        else:
            net_name = 'net%d'%net_id
            file_path = osp.join(save_dir, '%s_%s.pt'%(self.module_name, net_name))
        state_dict = self.get_snapshot(net_name)
        torch.save(state_dict, file_path)
    
    def load(self, load_dir, net_id=None):
        if self.ensemble_size is None or net_id is None:
            net_name = ''
            file_path = osp.join(load_dir, '%s.pt'%self.module_name)
        else:
            net_name = 'net%d'%net_id
            file_path = osp.join(load_dir, '%s_%s.pt'%(self.module_name, net_name))
            if not osp.exists(file_path):
                file_path = osp.join(load_dir, '%s.pt'%self.module_name)
        loaded_state_dict = torch.load(file_path)
        self.load_snapshot(loaded_state_dict, net_name)

    def forward(self, x, output_layer=-1, **kwargs):
        if self.ensemble_size is None:
            max_output_dim  = x.dim()
        else:
            max_output_dim  = x.dim() + 1
        
        n_fc = len(self.fcs)
        if output_layer < 0:
            output_layer = n_fc + output_layer

        i = 0
        for fc,act_f in zip(self.fcs, self.activation_functions):
            x = fc(x, **kwargs)
            x = act_f(x)
            if i >= output_layer:
                break
            i += 1
        
        while x.dim() > max_output_dim:
            x = x.squeeze(0)
        return x

    def get_weight_decay(self, weight_decays):
        weight_decays = to_list(weight_decays, len(self.fcs))
        weight_decay_tensors = []
        for weight_decay, fc in zip(weight_decays, self.fcs):
            weight_decay_tensors.append(fc.get_weight_decay(weight_decay))
        return sum(weight_decay_tensors)

class NoisyMLP(MLP):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_layers=[128,128], 
                 ensemble_size=None, 
                 noise_type='gaussian', #'uniform', None 
                 output_noise_type='gaussian',
                 factorised=True,
                 output_factorised=True,
                 **mlp_kwargs
                 ):
        self.output_noise_type = output_noise_type
        self.noise_types = to_list(noise_type, len(hidden_layers))
        self.noise_types.append(output_noise_type)
        
        self.output_factorised = output_factorised
        self.factorised = to_list(factorised, len(hidden_layers))
        self.factorised.append(output_factorised)

        super(NoisyMLP, self).__init__(input_size, 
                                       output_size, 
                                       hidden_layers, 
                                       ensemble_size, 
                                       **mlp_kwargs)

    def _get_single_layer(self, i):
        noise_type = self.noise_types[i]
        if noise_type is None:
            return super(NoisyMLP, self)._get_single_layer(i)

        factorised = self.factorised[i]

        if self.ensemble_size is None:
            fc = NoisyLinear(self.layers[i],
                             self.layers[i+1],
                             noise_type=noise_type,
                             factorised=factorised,
                             which_nonlinearity=self.nonlinearities[i],
                             **self.fc_kwargs)
        else:
            fc = NoisyEnsembleLinear(self.layers[i], 
                                     self.layers[i+1], 
                                     self.ensemble_size,
                                     noise_type=noise_type,
                                     factorised=factorised,
                                     which_nonlinearity=self.nonlinearities[i],
                                     **self.fc_kwargs)
        return fc
