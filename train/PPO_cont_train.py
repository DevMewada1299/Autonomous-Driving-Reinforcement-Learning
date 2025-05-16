import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env, VecEnv
from stable_baselines3 import PPO
from agents.PPO_cont_agent import create_ppo_cont_agent
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

register(
    id='highway-custom-PPO-cont',
    entry_point='Custom_Env.Highway_env_continuos:highwayEnvContinuos',
)

env = make_vec_env("highway-custom-PPO-cont", n_envs=1)
vec_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

vec_env.reset()

checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path='../models/PPO_cont/PPO_cont_checkpoints/',
    name_prefix='ppo_cont_highway'
)

model = PPO.load("../models/PPO_cont/PPO_cont_highway.zip", env=vec_env)
# model = create_ppo_cont_agent(vec_env)
model.learn(total_timesteps=200000, callback=CheckpointCallback)
model.save("../models/PPO_cont/PPO_cont_highway.zip")
vec_env.save("../models/PPO_cont/vecnormalize.pkl")
env.close()

