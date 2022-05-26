import abc

class Collector(object, metaclass=abc.ABCMeta):
    def start_epoch(self, epoch=None):
        pass

    def end_epoch(self, epoch=None):
        pass

    def get_diagnostics(self):
        return {}

    @abc.abstractmethod
    def get_epoch_paths(self):
        pass
 

class PathCollector(Collector, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        pass


class StepCollector(Collector, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        pass
