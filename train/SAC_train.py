import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
from agents.SAC_agent import create_sac_agent
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import CheckpointCallback

register(
    id='highway-custom-PPO-cont',
    entry_point='Custom_Env.Highway_env_continuos:highwayEnvContinuos',
)

env = make_vec_env("highway-custom-PPO-cont", n_envs=1)

env.reset()

checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path='../models/SAC/SAC_checkpoints/',
    name_prefix='sac_highway'
)


# model = create_sac_agent(env)
model = SAC.load("../models/SAC/SAC_highway.zip",env)
model.learn(total_timesteps=200000, callback=CheckpointCallback)
model.save("../models/SAC/SAC_highway.zip")

env.close()

