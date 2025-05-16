from stable_baselines3 import PPO
import torch
import torch.nn as nn


# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

policy_kwargs = dict(
    net_arch=[64,64],
    activation_fn=torch.nn.ReLU,
    log_std_init=-0.5,
    ortho_init=True
)


def create_ppo_discrete_agent(env, log_dir="logs_PPO_discrete/"):
    model = PPO('MlpPolicy', env,
                learning_rate=0.001,
                n_steps=256,
                n_epochs=10,
                batch_size=64,
                tensorboard_log=log_dir,
                gamma=0.99,
                gae_lambda=0.95,
                seed=42,
                policy_kwargs=policy_kwargs,
                verbose=1,
                )

    return model