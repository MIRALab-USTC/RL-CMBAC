import gym
from gym import version, Wrapper
from mbrl.utils.logger import logger
from gym.wrappers.monitoring import video_recorder
from gym.utils import closer
import os, json, six
import numpy as np

from mbrl.environments.base_env import Env, SimpleEnv

FILE_PREFIX = 'mbrl-mujoco-video'

monitor_closer = closer.Closer()

class VideoEnv(Env):
    def __init__(self, env_name, directory=None, force=False):
        if directory is None: 
            directory = os.path.join(logger._snapshot_dir, 'videos')
        inner_env = SimpleEnv(env_name)
        Wrapper.__init__(self, inner_env)
        self.env = inner_env

        self.videos = []
        self.video_recorder = None
        self.enabled = False
        self.episode_id = 0
        self._monitor_id = None
        self.cur_video_name = None

        self._start(directory, force)

    @property
    def horizon(self):
        return self.env.horizon

    def set_video_name(self, video_name):
        self.cur_video_name = video_name

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._after_step()
        return observation, reward, done, info

    def reset(self, video_name=None, **kwargs):
        if video_name is not None:
            self.cur_video_name = video_name
        observation = self.env.reset(**kwargs)
        self._after_reset()
        return observation

    def _start(self, directory, force=False):
        """Start monitoring.
        Args:
            directory (str): A per-training run directory where to record stats.
            force (bool): Clear out existing training data from this directory (by deleting every file prefixed with "openaigym.").
        """
        if not os.path.exists(directory):
            logger.log('Creating monitor directory %s'%directory)
            if six.PY3:
                os.makedirs(directory, exist_ok=True)
            else:
                os.makedirs(directory)

        # Check on whether we need to clear anything
        if force:
            clear_monitor_files(directory)

        self._monitor_id = monitor_closer.register(self)

        self.enabled = True
        self.directory = os.path.abspath(directory)
        self.file_prefix = FILE_PREFIX
        self.file_infix = '{}'.format(self._monitor_id)

    def close(self):
        """Flush all monitor data to disk and close any open rending windows."""
        Wrapper.close(self)

        if not self.enabled:
            return
            
        if self.video_recorder is not None:
            self._close_video_recorder()

        # Stop tracking this for autoclose
        monitor_closer.unregister(self._monitor_id)
        self.enabled = False

    def _after_step(self):
        # Record video
        self.video_recorder.capture_frame()

    def _after_reset(self):
        if not self.enabled: return
        self.reset_video_recorder()
        # Bump *after* all reset activity has finished
        self.episode_id += 1

    def reset_video_recorder(self):
        # Close any existing video recorder
        if self.video_recorder:
            self._close_video_recorder()

        # Start recording the next video.
        if self.cur_video_name is not None:
            video_name = self.cur_video_name 
            self.cur_video_name = None
        else:
            video_name = 'episode{}'.format(self.episode_id) 
            
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=os.path.join(self.directory, '{}.{}.{}'.format(self.file_prefix, self.file_infix, video_name)),
            metadata={'video_name': video_name},
            enabled=True,
        )
        self.video_recorder.capture_frame()

    def _close_video_recorder(self):
        self.video_recorder.close()
        if self.video_recorder.functional:
            self.videos.append((self.video_recorder.path, self.video_recorder.metadata_path))

    def __del__(self):
        # Make sure we've closed up shop when garbage collecting
        self.close()

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith(FILE_PREFIX + '.')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return

    logger.log('Clearing %d monitor files from previous run (because force=True was provided)', len(files))
    for file in files:
        os.unlink(file)

