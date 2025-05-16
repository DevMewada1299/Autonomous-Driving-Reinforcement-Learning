from stable_baselines3 import TD3
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


from Custom_Env.highway_env_custom import highwayEnvCustom
from gymnasium.envs.registration import register
from Custom_Env.highway_env_custom import highwayEnvCustom
register(
    id='highway-custom-PPO-cont',
    entry_point='Custom_Env.Highway_env_continuos:highwayEnvContinuos',
)


vec_env = DummyVecEnv([lambda: gym.make("highway-custom-PPO-cont", render_mode = 'human')])


vec_env.reset()

model = TD3.load("../models/TD3/TD3_highway.zip")

for episode in range(1000):
    obs = vec_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()

vec_env.close()