import os
import torch
import ale_py  # Crucial for Gymnasium 1.0+ to register Atari ROMs
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

def main():
    print("Checking hardware availability...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    print("Initializing Vectorized Atari Environments...")
    # n_envs=8 spins up 8 games at once. Your 7700 XT will chew through this effortlessly.
    env = make_atari_env("ALE/Pong-v5", n_envs=8, seed=42, env_kwargs={"render_mode": None})
    
    # Stack 4 frames so the CNN can perceive motion/velocity
    env = VecFrameStack(env, n_stack=4)

    print("Building the Core CNN Model with PPO...")
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        device=device
    )

    training_steps = 200_000
    print(f"Starting training loop for {training_steps} total timesteps...")
    
    try:
        model.learn(total_timesteps=training_steps)
        print("Training complete!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")

    # Save the weights
    model_path = "ppo_pong_model"
    model.save(model_path)
    print(f"Model saved cleanly to {model_path}.zip")

if __name__ == "__main__":
    main()