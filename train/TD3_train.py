import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from agents.TD3_agent import create_td3_agent
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

register(
    id='highway-custom-PPO-cont',
    entry_point='Custom_Env.Highway_env_continuos:highwayEnvContinuos',
)


checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path='../models/TD3/TD3_checkpoints/',
    name_prefix='td3_highway'
)

env = make_vec_env("highway-custom-PPO-cont", n_envs=1)
vec_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

vec_env.reset()


model = create_td3_agent(vec_env)
model.learn(total_timesteps=200000, callback=CheckpointCallback)
model.save("../models/TD3/TD3_highway.zip")
vec_env.save("../models/TD3/vecnormalize_td3.pkl")

env.close()

