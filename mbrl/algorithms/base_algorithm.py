import abc
import gtimer as gt
from collections import OrderedDict

from mbrl.algorithms.utils import get_dict_of_items_from_config
from mbrl.utils.eval_util import get_generic_path_information
from mbrl.utils.logger import logger


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class RLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            item_dict_config={},
    ):
        gt.reset_root()
        # Get items and update algorithm with these items
        self.item_dict_config = item_dict_config
        self.item_dict = get_dict_of_items_from_config(item_dict_config)
        self.__dict__.update(self.item_dict)
        self._need_snapshot = []

    def _before_train(self):
        pass
    def _after_train(self):
        pass

    def train(self):     
        for epoch in gt.timed_for( range(-1, self.num_epochs+1),
                                   save_itrs=True,
                                 ):
            if epoch == -1:
                self._before_train()
            elif epoch == self.num_epochs:
                self._after_train()
            else:       
                self.train_epoch(epoch)

    def train_epoch(self, epoch):
        self.start_epoch(epoch)
        self._train_epoch(epoch)
        self.end_epoch(epoch)

    @abc.abstractmethod
    def _train_epoch(self, epoch):
        pass

    def start_epoch(self, epoch=None):
        for _, item in self.item_dict.items():
            if hasattr(item, 'start_epoch'):
                item.start_epoch(epoch)
        self._start_epoch(epoch)
        gt.stamp('starting epoch')

    def _start_epoch(self, epoch):
        pass

    def end_epoch(self, epoch=None):
        for _, item in self.item_dict.items():
            if hasattr(item, 'end_epoch'):
                item.end_epoch(epoch)
        self._end_epoch(epoch)
        gt.stamp('ending epoch')

        # Save log and snapshot
        if epoch is not None and epoch >= 0:
            snapshot = self.get_snapshot()
            logger.save_itr_params(epoch, snapshot)
            gt.stamp('saving')
            self.log_stats(epoch)

    def _end_epoch(self, epoch):
        pass

    def get_snapshot(self):
        snapshot = {}
        for item_name in self._need_snapshot:
            item = self.item_dict[item_name]
            if hasattr(item, 'get_snapshot'):
                print("save snapshot: %s"%item_name)
                snapshot[item_name] = {}
                for k, v in item.get_snapshot().items():
                    snapshot[item_name][k] = v
        return snapshot

    def log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        if hasattr(self, 'pool'):
            logger.record_dict(
                self.pool.get_diagnostics(),
                prefix='replay_pool/'
            )

        """
        Trainer
        """
        if hasattr(self, 'trainer'):
            logger.record_dict(
                self.trainer.get_diagnostics(),
                prefix='trainer/'
            )

        """
        Exploration
        """
        if hasattr(self, 'expl_env') and hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(),
                prefix='exploration/',
            )

        if hasattr(self, 'expl_collector'):
            expl_paths = self.expl_collector.get_epoch_paths()
            logger.record_dict(
                self.expl_collector.get_diagnostics(),
                prefix='exploration/'
            )
            logger.record_dict(
                get_generic_path_information(expl_paths),
                prefix="exploration/",
            )

        """
        Evaluation
        """
        if hasattr(self, 'eval_env') and hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_env.get_diagnostics(),
                prefix='evaluation/',
            )
            
        if hasattr(self, 'eval_collector'):
            logger.record_dict(
                self.eval_collector.get_diagnostics(),
                prefix='evaluation/',
            )
            eval_paths = self.eval_collector.get_epoch_paths()
            logger.record_dict(
                get_generic_path_information(eval_paths),
                prefix="evaluation/",
            )
        """
        Analyzer
        """
        if hasattr(self, 'analyzer'):
            logger.record_dict(
                self.analyzer.get_diagnostics(),
                prefix='analyzer/',
            )
        
        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        if hasattr(self, 'trainer'):
            for net in self.trainer.networks:
                net.train(mode)

    def to(self, device):
        for item_name, item in self.item_dict.items():
            if hasattr(item, 'to'):
                item.to(device)
