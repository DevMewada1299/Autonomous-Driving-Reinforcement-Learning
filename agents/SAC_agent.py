from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import torch


policy_kwargs = dict(
    net_arch=[256, 256],
    log_std_init=-0.5,
    activation_fn=torch.nn.ReLU
)

def create_sac_agent(env, log_dir="logs_SAC/"):
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
    )

    return model


