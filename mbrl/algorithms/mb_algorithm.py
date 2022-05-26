import abc
import gtimer as gt
from collections import OrderedDict

from mbrl.algorithms.batch_rl_algorithm import BatchRLAlgorithm
from mbrl.utils.eval_util import get_generic_path_information
from mbrl.utils.process import Progress, Silent, format_for_process
from mbrl.utils.misc_untils import combine_item
from mbrl.utils.logger import logger
import mbrl.torch_modules.utils as ptu
from mbrl.collectors.utils import rollout

import numpy as np
import torch
import copy
import pdb


class MBRLBatchAlgorithm(BatchRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            num_epochs,
            batch_size,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_train_loops_per_epoch,
            num_trains_per_train_loop,
            max_path_length=1000,
            min_num_steps_before_training=0,
            silent = False,
            record_video_freq=50,
            analyze_freq=1,
            real_data_ratio=0.1,
            train_model_freq=250,
            imagine_freq=250,
            env_name=None,
            save_data_freq=0,
            train_model_config={},
            item_dict_config={},
    ):
        super().__init__(
            num_epochs,
            batch_size,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_train_loops_per_epoch,
            num_trains_per_train_loop,
            max_path_length,
            min_num_steps_before_training,
            silent,
            record_video_freq,
            analyze_freq,
            item_dict_config
            )
        self.real_data_ratio = real_data_ratio
        self.train_model_config = train_model_config
        self.train_model_freq = train_model_freq
        self.imagine_freq = imagine_freq
        assert (imagine_freq % train_model_freq) == 0
        self.env_name = env_name
        self.save_data_freq = save_data_freq

        self._need_snapshot.append('model_trainer')

        o_shape = self.expl_env.observation_space.shape
        extra_fields = {
            'deltas': {
                'shape': o_shape,
                'type': np.float,
            },
        }
        self.pool.add_extra_fields(extra_fields)

        self.real_batch_size = int(batch_size * real_data_ratio)
        self.imagined_batch_size = batch_size - self.real_batch_size
        self.init_model = True
        self.model_diagnostics = {}

    def _train_model(self):
        self.model_diagnostics = self.model_trainer.train_with_pool(self.pool, **self.train_model_config)

    def _train_epoch(self, epoch):
        progress = self.progress_class(self.num_train_loops_per_epoch * self.num_trains_per_train_loop)
        for i in range(self.num_train_loops_per_epoch):
            if i % self.train_model_freq == 0:
                self.training_mode(True)
                self._train_model()
                self.training_mode(False)
                gt.stamp('training model', unique=False)
            if i % self.imagine_freq == 0:
                self.model_collector.imagine(epoch)
                gt.stamp('rollout', unique=False)

            self._sample(self.num_expl_steps_per_train_loop)
            gt.stamp('exploration sampling', unique=False)

            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                progress.update()
                params = self._train_batch()

                params['Model Train Loss'] = self.model_diagnostics['train_loss']
                params['Model Eval Loss'] = self.model_diagnostics['eval_loss']

                progress.set_description(format_for_process(params))
            self.training_mode(False)
            gt.stamp('training', unique=False)

        self.eval_collector.collect_new_paths(
            self.num_eval_steps_per_epoch,
            self.max_path_length,
            discard_incomplete_paths=True,
        )
        gt.stamp('evaluation sampling')
        progress.close()

    def _train_batch(self):
        real_batch = self.pool.random_batch(self.real_batch_size, without_keys=['deltas'])
        imagined_batch= self.imagined_data_pool.random_batch(self.imagined_batch_size)
        train_data = combine_item(real_batch, imagined_batch)
        params = self.trainer.train(train_data)
        return params

    def log_stats(self, epoch):
        logger.record_dict(
            self.model_diagnostics,
            prefix='model/'
        )
        logger.record_dict(
            self.imagined_data_pool.get_diagnostics(),
            prefix='model_pool/'
        )
        super().log_stats(epoch)

    def _end_epoch(self, epoch):
        if (
                self.analyze_freq > 0 and \
                    (
                        epoch % self.analyze_freq == 0 or \
                        epoch == self.num_epochs-1
                    ) and \
                hasattr(self, 'analyzer')
            ):
            self.analyzer.analyze()
        gt.stamp('analyze')
        
        if (
                self.record_video_freq > 0 and \
                (
                    epoch % self.record_video_freq == 0 or \
                    epoch == self.num_epochs-1
                ) and \
                hasattr(self, 'video_env')
            ):
            self.video_env.set_video_name("epoch{}".format(epoch))
            logger.log("rollout to save video...")
            rollout(self.video_env, self.eval_policy, max_path_length=self.max_path_length, use_tqdm=True)
        gt.stamp('save video', unique=False)
