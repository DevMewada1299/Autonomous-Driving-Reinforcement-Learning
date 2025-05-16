from stable_baselines3 import PPO
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import SubprocVecEnv




# Define policy architecture
policy_kwargs = dict(
    net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64]),  # Separate actor and critic networks
    activation_fn=torch.nn.ReLU,
    log_std_init=-0.5,  # Reduce exploration noise
    ortho_init=True
)

def create_ppo_cont_agent(env, log_dir="logs_PPO_cont/"):

    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,  # Lower learning rate for stability
        n_steps=2048,         # Increase rollout steps
        n_epochs=10,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,      # Standard PPO clipping
        ent_coef=0.01,       # Add entropy to encourage exploration
        vf_coef=0.3,         # Balance value function loss
        max_grad_norm=0.5,   # Clip gradients to prevent large updates
        tensorboard_log=log_dir,
        seed=42,
        policy_kwargs=policy_kwargs,
        verbose=1,
        normalize_advantage=True
    )

    return model