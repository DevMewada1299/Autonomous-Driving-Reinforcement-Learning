from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

from Custom_Env.highway_env_custom import highwayEnvCustom
from gymnasium.envs.registration import register
from Custom_Env.highway_env_custom import highwayEnvCustom
register(
    id='highway-custom-PPO',
    entry_point='Custom_Env.HighwayEnvCustomV2_PPO:HighwayEnvCustomV2_PPO',
)

vec_env = DummyVecEnv([lambda: gym.make("highway-custom-PPO", render_mode = 'human')])



vec_env.reset()

model = PPO.load("../models/PPO_discrete_highway/PPO_discrete_highway.zip")

# for episode in range(1000):
#     obs = vec_env.reset()
#     done = False
#     while not done:
#         action, _ = model.predict(obs)
#         obs, reward, done, info = vec_env.step(action)
#         vec_env.render()
#
# vec_env.close()

env = vec_env.envs[0].unwrapped

episode_rewards = []
episode_lengths = []
collision_count = 0
episode_speeds = []
min_speeds = []
max_speeds = []
lane_changes = []

for episode in range(10):
    obs = vec_env.reset()
    done = False
    total_reward = 0
    steps = 0
    speeds = []
    crashed = False
    last_lane = env.vehicle.lane_index[2]
    lane_change_count = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)
        # vec_env.render()  # Optional: Disable for faster testing

        speed = env.vehicle.speed * np.cos(env.vehicle.heading)
        speeds.append(speed)
        total_reward += reward[0]
        steps += 1

        if not crashed and info[0].get("crashed", False):
            crashed = True

        current_lane = env.vehicle.lane_index[2]
        if current_lane != last_lane:
            lane_change_count += 1
            last_lane = current_lane

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    collision_count += int(crashed)
    episode_speeds.append(np.mean(speeds))
    min_speeds.append(np.min(speeds))
    max_speeds.append(np.max(speeds))
    lane_changes.append(lane_change_count)
    if crashed:
        collision_count += 1

print(f"Average Reward: {np.mean(episode_rewards):.2f}")
print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
print(f"Collision Rate: {100 * collision_count / len(episode_rewards):.2f}%")
print(f"Average Speed: {np.mean(episode_speeds):.2f} m/s")
print(f"Min Speed (avg across episodes): {np.mean(min_speeds):.2f} m/s")
print(f"Max Speed (avg across episodes): {np.mean(max_speeds):.2f} m/s")
print(f"Average Lane Changes per Episode: {np.mean(lane_changes):.2f}")