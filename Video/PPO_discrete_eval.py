import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from Custom_Env.highway_env_custom import highwayEnvCustom
from gymnasium.envs.registration import register

register(
    id='highway-custom-PPO',
    entry_point='Custom_Env.HighwayEnvCustomV2_PPO:HighwayEnvCustomV2_PPO',
)


# Create base env with render_mode="rgb_array"
env = gym.make("highway-custom-PPO", render_mode="rgb_array")

# Wrap with RecordVideo BEFORE DummyVecEnv
env = RecordVideo(
    env,
    video_folder="./videos",
    episode_trigger=lambda episode_id: True,
    name_prefix="ppo_eval"
)

# Now you can evaluate without DummyVecEnv
model = PPO.load("../models/PPO_discrete_highway/PPO_discrete_highway.zip")

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)

env.close()