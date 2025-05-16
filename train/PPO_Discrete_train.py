import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

from agents.PPO_Discrete_agent import create_ppo_discrete_agent
from gymnasium.envs.registration import register

register(
    id='highway-custom-PPO',
    entry_point='Custom_Env.HighwayEnvCustomV2_PPO:HighwayEnvCustomV2_PPO',
)

env = make_vec_env("highway-custom-PPO", n_envs=1)

env.reset()

model = create_ppo_discrete_agent(env)
model.learn(total_timesteps=10000)
model.save("../models/PPO_discrete_highway/PPO_discrete_highway.zip")

env.close()

