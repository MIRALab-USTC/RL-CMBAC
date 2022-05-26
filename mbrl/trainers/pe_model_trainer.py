from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from os.path import join

import mbrl.torch_modules.utils as ptu
from mbrl.utils.eval_util import create_stats_ordered_dict
from mbrl.trainers.base_trainer import BatchTorchTrainer
from mbrl.pools.utils import split_dataset
from mbrl.utils.process import Progress, Silent, format_for_process
from mbrl.utils.logger import logger


class PEModelTrainer(BatchTorchTrainer):
    def __init__(
            self,
            env,
            model,
            lr=3e-4,
            weight_decay=5e-5,
            init_model_train_step=int(2e4),
            train_mode="sufficiently",
            optimizer_class='Adam',
            extra_loss_coefficient=0.05,
    ):
        super().__init__()
        if isinstance(optimizer_class, str):
            optimizer_class = eval('optim.'+optimizer_class)
        self.env = env
        self.model = model
        self.optimizer = optimizer_class(
            self.model.module.parameters(),
            lr=lr,
        )
        self.weight_decay = weight_decay
        self.statistics = OrderedDict()

        self.learn_reward = model.learn_reward
        self.learn_done = model.learn_done

        self._n_train_steps_total = 0
        self.init_model_train_step = int(init_model_train_step)
        self.train_mode = train_mode
        self.extra_loss_coefficient = extra_loss_coefficient


    def compute_ensemble_loss(self, o, a, deltas, r, d):    
        log_prob, log_std = self.model.log_prob(o, a, deltas, r, d)
        ensemble_loss = torch.mean(log_prob, dim=1)
        ensemble_loss = 0-torch.mean(ensemble_loss, dim=-1)
        return ensemble_loss, log_std

    def train_from_torch_batch(self, batch):
        o = batch['observations']
        a = batch['actions']
        deltas = batch['deltas']
        r = batch['rewards']
        d = batch['terminals']

        ensemble_loss, log_std = self.compute_ensemble_loss(o,a,deltas,r,d)
        original_loss = torch.mean(ensemble_loss) 
        loss = original_loss + self.model.module.get_weight_decay(self.weight_decay)
        extra_loss, diagnostics_extra_loss = self.model.module.get_extra_loss(self.extra_loss_coefficient)
        loss = loss + extra_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        diagnostics = OrderedDict()
        ensembel_loss_np = ptu.get_numpy(ensemble_loss)
        for i,l in enumerate(ensembel_loss_np):
            diagnostics['net_%d/train_loss'%i] = l
        diagnostics['train_loss'] = original_loss.item()
        diagnostics.update(
            create_stats_ordered_dict(
                "log_std",
                ptu.get_numpy(log_std),
            )
        )
        diagnostics.update(diagnostics_extra_loss)
        self._n_train_steps_total += 1
        return diagnostics

    def eval_with_dataset(self, dataset):
        dataset = ptu.np_to_pytorch_batch(dataset)
        o = dataset['observations']
        a = dataset['actions']
        deltas = dataset['deltas']
        r = dataset['rewards']
        d = dataset['terminals']
        ensemble_loss, _ = self.compute_ensemble_loss(o,a,deltas,r,d)
        loss = torch.mean(ensemble_loss)

        diagnostics = OrderedDict()
        ensemble_loss_np = ptu.get_numpy(ensemble_loss)
        for i,l in enumerate(ensemble_loss_np):
            diagnostics['net_%d/eval_loss'%i] = l
        diagnostics['eval_loss'] = loss.item()
        
        return diagnostics, list(ensemble_loss_np)

    def set_mean_std(self, mean_std_dict=None):
        model = self.model
        if mean_std_dict is None:
            mean_std_dict = self.mean_std_dict
        else:
            self.mean_std_dict = mean_std_dict
        if model.normalize_obs:
            model.obs_processor.set_mean_std_np(*mean_std_dict['observations'])
        if model.normalize_action:
            model.action_processor.set_mean_std_np(*mean_std_dict['actions'])
        if model.normalize_delta:
            model.delta_processor.set_mean_std_np(*mean_std_dict['deltas'])
        if model.normalize_reward:
            model.reward_processor.set_mean_std_np(*mean_std_dict['rewards'])

    def train_with_pool(
                    self,
                    pool,
                    max_step=2000,
                    **train_kwargs):        
        # train_mode: sufficiently or n_step
        temp_data = pool.get_unprocessed_data('compute_delta', ['observations', 'next_observations'])
        deltas = temp_data['next_observations'] - temp_data['observations']
        pool.update_single_extra_field('deltas', deltas)
        pool.update_process_flag('compute_delta', len(deltas))
        self.set_mean_std(pool.get_mean_std())

        train_mode = self.train_mode
        if self._n_train_steps_total == 0:
            self.init_model = False
            max_step = self.init_model_train_step
            train_mode = "sufficiently"
        else:
            max_step = int(max_step)

        if train_mode == "sufficiently":
            return self.sufficiently_train(pool, max_step, **train_kwargs)
        elif train_mode == "n_step":
            return self.n_step_train(pool, max_step, **train_kwargs)

    def split_dataset(self, pool, valid_ratio, max_valid, resample):
        ensemble_size = self.model.ensemble_size
        dataset = pool.get_data()
        train_dataset, eval_dataset, train_length, eval_length = split_dataset(dataset, valid_ratio, max_valid)
        self.train_dataset, self.eval_dataset = train_dataset, eval_dataset
        self.train_length, self.eval_length = train_length, eval_length

        if resample:
            self.ensemble_index = np.random.randint(train_length, size=(ensemble_size, train_length))
        else:
            ensemble_index = np.arange(train_length)[None]
            self.ensemble_index = np.tile(ensemble_index, (ensemble_size,1))

    def save_dataset(self, save_dir=None):
        save_dic = {
            "train_dataset": self.train_dataset,
            "eval_dataset": self.eval_dataset,
            "train_length": self.train_length,
            "eval_length": self.eval_length,
            "ensemble_index": self.ensemble_index,
            "mean_std_dict": self.mean_std_dict
        }
        if save_dir == None:
            save_dir = logger._snapshot_dir
        path = join(save_dir, 'pe_model_dataset.npy')
        np.save(path, save_dic, allow_pickle=True)
    
    
    def load_dataset(self, load_dir=None):
        if load_dir == None:
            load_dir = logger._snapshot_dir
        path = join(load_dir, 'pe_model_dataset.npy')
        load_dic = np.load(path, allow_pickle=True).item()
        self.__dict__.update(load_dic)
            
    
    def _get_batch_from_dataset(self, dataset, index):
        batch = {}
        for k in dataset:
            batch[k] = dataset[k][index]
        return batch
    
    def sufficiently_train(
                    self,
                    pool=None,
                    max_step=2000, 
                    valid_ratio=0.2,
                    max_valid=5000,   
                    resample=True,             
                    batch_size=256, 
                    report_freq=20,
                    max_not_improve=15,
                    silent=False,
                    load_dataset_dir=None,
                    **useless_kwargs):
        ### test_code
        model = self.model
        if pool is not None:
            self.split_dataset(pool, valid_ratio, max_valid, resample)
        else:
            assert load_dataset_dir is not None
            self.load_dataset(load_dataset_dir)
            self.set_mean_std()

        train_dataset, eval_dataset = self.train_dataset, self.eval_dataset
        train_length = self.train_length
        ensemble_index = self.ensemble_index
        
        progress_class = Silent if silent else Progress
        progress = progress_class(int(max_step))
        total_train_step = 0
        while True:
            for j in range(0,train_length,batch_size):
                progress.update()
                ind = ensemble_index[:, j:j+batch_size]
                train_batch = self._get_batch_from_dataset(train_dataset, ind)
                train_stat = self.train_from_numpy_batch(train_batch)

                if total_train_step % report_freq == 0:
                    eval_stat, eval_loss = self.eval_with_dataset(eval_dataset)

                    if total_train_step == 0:
                        min_loss = eval_loss
                        not_improve = []
                        for i,_ in enumerate(eval_loss):
                            model.save(net_id=i)
                            not_improve.append(0) 
                    else:
                        for i,l in enumerate(eval_loss):
                            if l < min_loss[i]:
                                min_loss[i] = l
                                not_improve[i] = 0
                                model.save(net_id=i)
                            else:
                                not_improve[i] += 1
                    continue_training = False
                    for i,n in enumerate(not_improve):
                        eval_stat['net_%d/not_improve'%i] = n
                        #logger.tb_add_scalar(f"net_{i}/train_loss", train_stat[f"net_{i}/train_loss"], total_train_step)
                        #logger.tb_add_scalar(f"net_{i}/eval_loss", eval_stat[f"net_{i}/eval_loss"], total_train_step)
                        if n < max_not_improve:
                            continue_training = True

                    #logger.tb_add_scalar("model/train_loss", train_stat["train_loss"], total_train_step)
                    #logger.tb_add_scalar("model/eval_loss", eval_stat["eval_loss"], total_train_step)
                    #for k in train_stat.keys():
                    #    if "std" in k:
                    #        logger.tb_add_scalar(k, train_stat[k], total_train_step)

                stat = OrderedDict( list(train_stat.items()) + list(eval_stat.items()) )
                progress.set_description(format_for_process(stat))

                total_train_step += 1
                if total_train_step >= max_step:
                    continue_training = False
                
                if not continue_training:
                    #logger.tb_flush()
                    #logger.tb_close()
                    break
            if not continue_training:
                #logger.tb_flush()
                #logger.tb_close()
                break
        
        for i in range(len(eval_loss)):
            model.load(net_id=i)

        progress.close()
        eval_stat, eval_loss = self.eval_with_dataset(eval_dataset)
        model.remember_loss(eval_loss)
        self.statistics.update(train_stat)
        self.statistics.update(eval_stat)
        self.statistics['train_step'] = total_train_step
        self.min_loss = eval_loss
        return self.statistics
        ### test_code
    
    def n_step_train(
            self,
            pool,
            max_step=4, 
            valid_ratio=0.2,
            max_valid=5000,   
            resample=True,             
            batch_size=256,   
            report_freq=20,
            reload_freq=1000,
            **useless_kwargs
            ):
        for _ in range(max_step):
            ind_ind = np.random.choice(self.train_length, (batch_size,), replace=False)
            ind = self.ensemble_index[:, ind_ind]
            train_batch = self._get_batch_from_dataset(self.train_dataset, ind)
            train_stat = self.train_from_numpy_batch(train_batch)
            
            if self._n_train_steps_total % report_freq:
                eval_stat, eval_loss = self.eval_with_dataset(self.eval_dataset)
                self.statistics.update(eval_stat)
                for i,l in enumerate(eval_loss):
                    if l < self.min_loss[i]:
                        self.min_loss[i] = l
                        self.model.save(net_id=i)
            if self._n_train_steps_total % reload_freq:
                for i in range(len(self.min_loss)):
                    self.model.load(net_id=i)
                self.split_dataset(pool, valid_ratio, max_valid, resample)

        self.statistics.update(train_stat)
        return self.statistics
            

    def get_diagnostics(self):
        return self.statistics

    def end_epoch(self, epoch):
        pass

    @property
    def networks(self):
        return [self.model]

    def get_snapshot(self):
        return dict(model=self.model)

