from stable_baselines3 import A2C
import torch
import torch.nn as nn

def create_A2C_agent(env, log_dir="logs_A2C_discrete/"):
    model = A2C(
        "MlpPolicy",
        env,
        n_steps=128,
        gamma=0.9,
        gae_lambda=0.95,
        ent_coef=0.001,
        learning_rate=1e-4,
        verbose=1,
        tensorboard_log=log_dir,
    )

    return model