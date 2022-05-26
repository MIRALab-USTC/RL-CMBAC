
from gym.envs.registration import register

register(
    id='MultiGoal2DRandomReset-v0',
    entry_point='mbrl.environments.our_envs.multi_goal:MultiGoal2DEnv',
    kwargs={
        'random_reset':True,
        'goals':[[5,0], [-5,0]],
        'bound': [8,5]
    },
    max_episode_steps=30,
)
register(
    id='MultiGoal2D-v0',
    entry_point='mbrl.environments.our_envs.multi_goal:MultiGoal2DEnv',
    kwargs={
        'random_reset':False,
        'goals':[[5,0], [-5,0]],
        'bound': [8,5]
    },
    max_episode_steps=30,
)
register(
    id='Goal2D-v0',
    entry_point='mbrl.environments.our_envs.multi_goal:MultiGoal2DEnv',
    kwargs={
        'random_reset':True,
        'goals':[0,0],
        'bound': [2,2]
    },
    max_episode_steps=30,
)

register(
    id='AntTruncatedObs-v2',
    entry_point='mbrl.environments.our_envs.truncated_ant:AntTruncatedObsEnv',
    max_episode_steps=1000,
)
register(
    id='HumanoidTruncatedObs-v2',
    entry_point='mbrl.environments.our_envs.truncated_humanoid:HumanoidTruncatedObsEnv',
    max_episode_steps=1000,
)


env_name_to_gym_registry_dict = {
    "half_cheetah": "HalfCheetah-v2",
    "cheetah": "HalfCheetah-v2",
    "swimmer": "Swimmer-v2",
    "ant": "Ant-v2",
    "mb_ant": "AntTruncatedObs-v2",
    "hopper": "Hopper-v2",
    "walker2d": "Walker2d-v2",
    "humanoid": "Humanoid-v2",
    "mb_humanoid": "HumanoidTruncatedObs-v2",
    "2dpointer": "MultiGoal2D-v0",
    "inverted_pendulum": "InvertedPendulum-v2",
    "2dpoint": "Goal2D-v0"
}

short_name_dict = {
    "half_cheetah": "cheetah",
    "HalfCheetah-v2": "cheetah",
    "Swimmer-v2": "swimmer",
    "Ant-v2": "ant",
    "AntTruncatedObs-v2": "mb_ant",
    "Hopper-v2": "hopper",
    "Walker2d-v2": "walker2d",
    "Humanoid-v2": "humanoid",
    "HumanoidTruncatedObs-v2": "mb_humanoid",
    "MultiGoal2D-v0": "2dpointer",
    "Goal2D-v0": "2dpoint"
}

def get_short_name(env_name):
    return short_name_dict.get(env_name, env_name)
