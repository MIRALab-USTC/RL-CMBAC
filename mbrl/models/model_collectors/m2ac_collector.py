import numpy as np
from mbrl.models.model_collectors.mbpo_collector import MBPOCollector
from mbrl.global_config import *
from mbrl.models.utils import square_wass2_gaussian
import pdb

def get_scheduled_value(current, schedule):
    if type(schedule) in [float, int]:
        return schedule
    start, end, min_value, max_value = schedule
    ratio = (current - start) / (end - start) 
    value = (ratio * (max_value - min_value)) + min_value
    value = max(min(value, max_value), min_value)
    return value

class M2ACCollector(MBPOCollector):
    def __init__(self, 
                model, 
                policy, 
                pool,
                imagined_data_pool, 
                depth_schdule=1,
                number_sample=int(1e5),
                save_n=20,
                bathch_size=None,
                hard_stop=True,
                rollout_terminal_state=False,
                step_kwargs={
                    "return_distribution": True 
                },
                u_type='mean w2',
                penalize=0.001,
                discard_lambda= 'lambda d,e: 0.5',
        ):
        # uncertainty type: 
        # the input of discard_lambda: d(depth), e(epoch)
        super().__init__(
                model, 
                policy, 
                pool,
                imagined_data_pool, 
                depth_schdule,
                number_sample,
                save_n,
                bathch_size,
                hard_stop,
                rollout_terminal_state,
                step_kwargs,
        )
        self.u_type = u_type,
        self.penalize = penalize
        self.discard_lambda = eval(discard_lambda)
        self._add_fields()

    def _add_fields(self):
        # ensemble_size = self.model.elite_number
        ensemble_size = self.model.ensemble_size
        o_shape = self.model.observation_shape
        a_shape = self.model.processed_action_shape
        learn_reward = self.model.learn_reward

        self.extra_fields = {
            'next_obs_mean': {
                'shape': o_shape,
                'type': np.float,
            },
            'next_obs_std': {
                'shape': o_shape,
                'type': np.float,
            },
            'ensemble_next_obs_mean': {
                'shape': (ensemble_size, *o_shape),
                'type': np.float,
            },
            'ensemble_next_obs_std': {
                'shape': (ensemble_size, *o_shape),
                'type': np.float,
            },
        }
        if learn_reward:
            self.extra_fields.update(
                {
                    'reward_mean': {
                        'shape': (1,),
                        'type': np.float,
                    },
                    'reward_std': {
                        'shape': (1,),
                        'type': np.float,
                    },
                    'ensemble_reward_mean': {
                        'shape': (ensemble_size, 1),
                        'type': np.float,
                    },
                    'ensemble_reward_std': {
                        'shape': (ensemble_size, 1),
                        'type': np.float,
                    },
                }
            )

        self.imagined_data_pool.add_extra_fields(self.extra_fields)


    def _get_live_ind(self, stop, info):
        if self.hard_stop:
            ind = (stop<0.5)
        else:
            temp_x = np.random.rand(*stop.shape)
            ind = (stop < temp_x)
        return ind.flatten()
    
    def _comput_uncertainty(self, info):
        u_type = self.u_type.split()
        m1 = info['next_obs_mean']
        std1 = info['next_obs_std']
        m2 = info['ensemble_next_obs_mean']
        std2 = info['ensemble_next_obs_std']

        if 'reward_mean' in info:
            r_m1 = info['reward_mean']
            r_std1 = info["reward_std"]
            m1 = np.concatenate([m1,r_m1],-1)
            std1 = np.concatenate([std1,r_std1],-1)

            r_m2 = info['ensemble_reward_mean']
            r_std2 = info['ensemble_reward_std']
            m2 = np.concatenate([m2,r_m2],-1)
            std2 = np.concatenate([std2,r_std2],-1)
        assert (m2.shape == std2.shape and m2.dim() == 3)

        if u_type[0] in ['mean', 'max']:
            if u_type[1] == 'w2':
                d = square_wass2_gaussian(m1, std1, m2, std2)
                if u_type[0] == 'mean':
                    u = d.mean(-1)
                else:
                    u = d.max(-1)
            else:
                raise NotImplementedError

        elif u_type[0] == 'OvR':
                raise NotImplementedError


    def add_sample(self, stop, o, a, next_o, r, d, info, depth, cur_epoch):
        samples = {}
        ind = self._get_live_ind(stop, info)
        samples['observations'] = o[ind]
        samples['actions'] = a[ind]
        samples['next_observations'] = next_o[ind]
        samples['rewards'] = r[ind]
        samples['terminals'] = d[ind]
        self.imagined_data_pool.add_samples(samples)
        for k in self.extra_fields:
            v = info[k][ind]
            self.imagined_data_pool.update_single_extra_field(k, v)

        if self.rollout_terminal_state:
            return next_o, d
        else:
            ind = self._get_live_ind(d, info)
            return next_o[ind], d[ind]