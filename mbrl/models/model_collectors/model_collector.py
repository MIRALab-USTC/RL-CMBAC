import abc

class ModelCollector(object, metaclass=abc.ABCMeta):
    def __init__(self, model, policy, pool, batch_size=None):
        raise NotImplementedError

    @abc.abstractmethod
    def imagine(self, states):
        pass