from stable_baselines3 import DQN
import torch
import torch.nn as nn


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

policy_kwargs = dict(
    net_arch=[128, 128, 64],
    activation_fn=nn.ReLU
)

def create_dqn_agent(env, log_dir="logs/"):
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=100,
        batch_size=32,
        tau=1.0,
        train_freq=4,
        target_update_interval=100,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        tensorboard_log=log_dir,
        device=device
    )

    return model