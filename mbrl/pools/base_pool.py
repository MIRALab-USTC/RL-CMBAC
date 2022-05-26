import abc
class Pool(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, env):
        pass

    @abc.abstractmethod
    def add_samples(self, samples):
        pass

    @abc.abstractmethod
    def add_paths(self, paths):
        pass
    
    @abc.abstractmethod
    def random_batch(self, batch_size):
        pass

    def get_diagnostics(self):
        return {}
