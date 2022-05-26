import numpy as np
from mbrl.models.model_collectors.mbpo_collector import MBPOCollector
from mbrl.models.utils import kl_gaussian
from mbrl.global_config import *
import pdb

MAX_MU = 100
MIN_MU = -100

def get_scheduled_value(current, schedule):
    if type(schedule) in [float, int]:
        return schedule
    start, end, min_value, max_value = schedule
    ratio = (current - start) / (end - start) 
    value = (ratio * (max_value - min_value)) + min_value
    value = max(min(value, max_value), min_value)
    return value

class MOPOCollector(MBPOCollector):
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
                u_type='mopo',
                penalize=0.001
        ):
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
        self.u_type = u_type
        self.penalize = penalize
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
            'model_uncertainty': {
                'shape': (1,),
                'type': np.float,
            }
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

    def _comput_uncertainty(self, info):
        m1 = info['next_obs_mean']
        std1 = info['next_obs_std']
        m2 = info['ensemble_next_obs_mean'] # (batch_size, ensemble_size, o_size)
        std2 = info['ensemble_next_obs_std']
        indexes = info['sample_indices'] # (batch_size,)
        assert indexes.shape[0] == m2.shape[0]
        batch_size = m2.shape[0]
        ensemble_size = m2.shape[1]
        n_arange = np.arange(batch_size)
        r_indexes = np.tile(np.arange(ensemble_size), batch_size).reshape(batch_size,-1)
        r_indexes = np.array([np.setdiff1d(r_indexes[i], indexes[i]) for i in range(batch_size)])

        if 'reward_mean' in info:
            r_m1 = info['reward_mean']
            r_std1 = info["reward_std"]
            m1 = np.concatenate([m1,r_m1],-1)
            std1 = np.concatenate([std1,r_std1],-1)

            r_m2 = info['ensemble_reward_mean']
            r_std2 = info['ensemble_reward_std']
            m2 = np.concatenate([m2,r_m2],-1)
            std2 = np.concatenate([std2,r_std2],-1)
        assert (m2.shape == std2.shape and m2.ndim == 3)

        if self.u_type == 'mopo':
            m_u = np.linalg.norm(std2, axis=-1)
            m_u = np.max(m_u, axis=-1, keepdims=True)
        elif self.u_type == 'morel':
            raise NotImplementedError
        elif self.u_type == 'm2ac':
            m_k = m2[n_arange,indexes]
            std_k = std2[n_arange,indexes]
            var_k = std_k ** 2
            m_r = [m2[n_arange, r_indexes[:,i]] for i in range(r_indexes.shape[1])]
            std_r = [std2[n_arange, r_indexes[:,i]] for i in range(r_indexes.shape[1])]
            m_r = np.stack(m_r, axis=1)
            std_r = np.stack(std_r, axis=1)
            m_r_mean = np.mean(m_r, axis=-2)
            var_r_mean = np.mean(np.square(std_r) + np.square(m_r), axis=-2) - np.square(m_r_mean)
            m_u = kl_gaussian(m_k, var_k, m_r_mean, var_r_mean)
            m_u = np.clip(m_u, MIN_MU, MAX_MU)
        else:
            raise NotImplementedError

        return m_u

    def add_sample(self, stop, o, a, next_o, r, d, info, depth, cur_epoch):
        samples = {}
        ind = self._get_live_ind(stop)
        samples['observations'] = o[ind]
        samples['actions'] = a[ind]
        samples['next_observations'] = next_o[ind]
        samples['terminals'] = d[ind]
        m_u = self._comput_uncertainty(info)
        info['model_uncertainty'] = m_u
        if self.penalize is not None:
            samples['rewards'] = r[ind] - self.penalize * m_u[ind]
        else:
            samples['rewards'] = r[ind]
        self.imagined_data_pool.add_samples(samples)
        for k in self.extra_fields:
            v = info[k][ind]
            self.imagined_data_pool.update_single_extra_field(k, v)
        if self.rollout_terminal_state:
            return next_o, d
        else:
            ind = self._get_live_ind(d)
            return next_o[ind], d[ind]
