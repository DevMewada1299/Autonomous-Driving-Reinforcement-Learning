from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium.envs.registration import register
import torch
import numpy as np

register(
    id='highway-custom-PPO-cont',
    entry_point='Custom_Env.Highway_env_continuos:highwayEnvContinuos',
)
# Define action noise for exploration
env = make_vec_env("highway-custom-PPO-cont", n_envs=1)
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))

# Policy architecture
policy_kwargs = dict(
    net_arch=[400, 300],
    activation_fn=torch.nn.ReLU,
)

def create_td3_agent(env, log_dir="logs_TD3/"):
    model = TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,               # More stable than 3e-4 in many continuous envs
        buffer_size=1_000_000,           # Larger buffer
        learning_starts=10_000,          # Delay updates until decent transitions are collected
        batch_size=256,
        tau=0.005,                       # More aggressive soft update
        gamma=0.99,
        train_freq=(1, "step"),          # Every step
        gradient_steps=1,
        policy_delay=2,                  # Standard for TD3
        target_policy_noise=0.2,         # Add noise to target actions
        target_noise_clip=0.5,           # Clip the target action noise
        action_noise=action_noise,       # Add exploration noise
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,

    )

    return model