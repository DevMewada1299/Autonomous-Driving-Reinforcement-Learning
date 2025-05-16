import gymnasium as gym
from agents.DQN_agent import create_dqn_agent
from gymnasium.envs.registration import register

register(
    id='highway-custom-v0',
    entry_point='Custom_Env.highway_env_custom:highwayEnvCustom',
)

env = gym.make("highway-custom-v0")

env.reset()

model = create_dqn_agent(env)
model.learn(total_timesteps=250000)
model.save("../models/DQN_highway/DQN_highway.zip")

env.close()

