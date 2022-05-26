import numpy as np
from mbrl.models.model_collectors.model_collector import ModelCollector
from mbrl.global_config import *
import pdb

def get_scheduled_value(current, schedule):
    if type(schedule) in [float, int]:
        return schedule
    start, end, min_value, max_value = schedule
    ratio = (current - start) / (end - start) 
    value = (ratio * (max_value - min_value)) + min_value
    value = max(min(value, max_value), min_value)
    return value

class MBPOCollector(ModelCollector):
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
                step_kwargs={},
        ):
        self.model = model
        self.policy = policy
        self.pool = pool
        self.imagined_data_pool = imagined_data_pool
        self.depth_schdule = depth_schdule
        self.number_sample = int(number_sample)
        self.save_n = save_n
        self.batch_size = bathch_size if bathch_size is None else int(bathch_size)
        self.step_kwargs = step_kwargs
        self.hard_stop = hard_stop
        self.rollout_terminal_state = rollout_terminal_state

        self._n_image_times_total = 0

    def imagine(self, cur_epoch):
        states = self.get_start_states()
        depth = self.cur_depth = self.get_depth(cur_epoch)
        self.resize_pool(cur_epoch)
        
        batch_size = self.batch_size
        number_sample = self.number_sample
        if batch_size is None:
            batch_size = number_sample

        for i in range(0, number_sample, batch_size):
            cur_size = min(batch_size, number_sample-i)
            o=states[i:i+cur_size]
            stop = np.zeros((cur_size,1))
            for j in range(depth):
                a = self.policy.action_np(o, return_info=False)
                next_o, r, d, info = self.model.step_np(o, a, return_info=True, **self.step_kwargs)
                d = stop + d - stop * d # stop or d
                o, stop = self.add_sample(stop,o,a,next_o,r,d,info,j,cur_epoch)
                if len(stop) == 0 or np.sum(stop) == len(stop):
                    break

        self._n_image_times_total += 1

    def get_start_states(self):
        data = self.pool.random_batch(int(self.number_sample), keys=['observations'])
        return data['observations']

    def _get_live_ind(self, stop):
        if self.hard_stop:
            ind = (stop<0.5).flatten()
        else:
            temp_x = np.random.rand(*stop.shape)
            ind = (stop < temp_x).flatten()
        return ind

    def add_sample(self, stop, o, a, next_o, r, d, info, depth, cur_epoch):
        samples = {}
        ind = self._get_live_ind(stop)
        samples['observations'] = o[ind]
        samples['actions'] = a[ind]
        samples['next_observations'] = next_o[ind]
        samples['rewards'] = r[ind]
        samples['terminals'] = d[ind]
        self.imagined_data_pool.add_samples(samples)
        if self.rollout_terminal_state:
            return next_o, d
        else:
            ind = self._get_live_ind(d)
            return next_o[ind], d[ind]
        
    def get_depth(self, cur_epoch):
        self.depth = int(get_scheduled_value(cur_epoch, self.depth_schdule))
        return self.depth

    def resize_pool(self, cur_epoch):
        size = int(self.depth * self.number_sample * self.save_n)
        if size != self.imagined_data_pool.max_size:
            self.imagined_data_pool.resize(size)

