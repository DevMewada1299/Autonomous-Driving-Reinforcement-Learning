import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC
from Custom_Env.highway_env_custom import highwayEnvCustom
from gymnasium.envs.registration import register

register(
    id='highway-custom-PPO-cont',
    entry_point='Custom_Env.Highway_env_continuos:highwayEnvContinuos',
)

# Create base env with render_mode="rgb_array"
env = gym.make("highway-custom-PPO-cont", render_mode="rgb_array")

# Wrap with RecordVideo BEFORE DummyVecEnv
env = RecordVideo(
    env,
    video_folder="./videos",
    episode_trigger=lambda episode_id: True,
    name_prefix="sac_eval"
)

# Now you can evaluate without DummyVecEnv
model = SAC.load("../models/SAC/SAC_highway.zip")

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)

env.close()