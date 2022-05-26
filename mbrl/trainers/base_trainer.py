import abc
from collections import OrderedDict
from typing import Iterable
from torch import nn
import mbrl.torch_modules.utils as ptu


class Trainer(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self, data):
        pass

    def start_epoch(self, epoch):
        pass

    def end_epoch(self, epoch):
        pass

    def get_snapshot(self):
        return {}

    def get_diagnostics(self):
        return {}

class BatchTorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    def train(self, data):
        return self.train_from_numpy_batch(data)

    def train_from_numpy_batch(self, np_batch):
        self._num_train_steps += 1
        batch = ptu.np_to_pytorch_batch(np_batch)
        return self.train_from_torch_batch(batch)

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps),
        ])

    @abc.abstractmethod
    def train_from_torch_batch(self, batch):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass


