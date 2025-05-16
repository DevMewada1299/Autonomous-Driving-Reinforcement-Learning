import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C
from agents.A2C_agent import create_A2C_agent
from agents.PPO_Discrete_agent import create_ppo_discrete_agent
from gymnasium.envs.registration import register
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class Logger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.iterations = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Track reward manually in case logger doesn't capture it
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True

    def _on_rollout_end(self) -> None:
        self.iterations += 1
        time_elapsed = time.time() - self.start_time
        fps = int(self.num_timesteps / (time_elapsed + 1e-8))

        # Get training stats
        policy_loss = self.logger.name_to_value.get("train/policy_loss")
        value_loss = self.logger.name_to_value.get("train/value_loss")
        entropy = self.logger.name_to_value.get("train/entropy_loss")

        # Prefer logger value, fallback to manual reward tracking
        ep_rew_mean = self.logger.name_to_value.get("rollout/ep_rew_mean")
        if ep_rew_mean is None and self.episode_rewards:
            ep_rew_mean = np.mean(self.episode_rewards[-10:])

        ep_len_mean = self.logger.name_to_value.get("rollout/ep_len_mean")
        if ep_len_mean is None and self.episode_lengths:
            ep_len_mean = np.mean(self.episode_lengths[-10:])

        print("\n-----------------------------------------")
        print(f"| iteration               | {self.iterations:<6}")
        print(f"| total timesteps         | {self.num_timesteps:<6}")
        print(f"| time elapsed            | {time_elapsed:.2f}s")
        print(f"| fps                     | {fps}")
        if ep_len_mean is not None:
            print(f"| ep_len_mean             | {ep_len_mean:.2f}")
        if ep_rew_mean is not None:
            print(f"| ep_rew_mean             | {ep_rew_mean:.2f}")
        if policy_loss is not None:
            print(f"| policy_loss             | {policy_loss:.5f}")
        if value_loss is not None:
            print(f"| value_loss              | {value_loss:.5f}")
        if entropy is not None:
            print(f"| entropy_loss            | {entropy:.5f}")
        print("-----------------------------------------\n")


register(
    id='highway-custom-v0',
    entry_point='Custom_Env.highway_env_custom:highwayEnvCustom',
)


env = make_vec_env('highway-custom-v0', n_envs=1)

env.reset()


model = A2C.load("../models/A2C_discrete/A2C_discrete.zip", env=env)
model.learn(total_timesteps=40000, callback=Logger())
model.save("../models/A2C_discrete/A2C_discrete.zip")

env.close()

