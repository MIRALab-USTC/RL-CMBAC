import abc
import mbrl.torch_modules.utils as ptu

class Value(object, metaclass=abc.ABCMeta):
    def __init__(self, env):
        self.action_shape = env.action_space.shape
        self.observation_shape = env.observation_space.shape

    def save(self, save_dir=None):
        raise NotImplementedError
     
    def load(self, load_dir=None):
        raise NotImplementedError

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}


class StateValue(Value, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def value(self, obs, return_info=True):
        pass

    def value_np(self, obs, return_info=True, **kwargs):
        obs = ptu.from_numpy(obs)
        if return_info:
            value, info = self.value(obs, return_info=return_info, **kwargs)
            value = ptu.get_numpy(value)
            info = ptu.torch_to_np_info(info)
            return value, info
        else:
            return ptu.get_numpy(self.value(obs, return_info=return_info, **kwargs))


class QValue(Value, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def value(self, obs, action, return_info=True):
        pass
    
    def value_np(self, obs, action, return_info=True, **kwargs):
        obs = ptu.from_numpy(obs)
        action = ptu.from_numpy(action)
        if return_info:
            value, info = self.value(obs, action, return_info=return_info, **kwargs)
            if value is not None:
                value = ptu.get_numpy(value)
            info = ptu.torch_to_np_info(info)
            return value, info
        else:
            return ptu.get_numpy(self.value(obs, action, return_info=return_info, **kwargs))

