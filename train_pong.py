import os
import sys
import torch
import ale_py  # Crucial for Gymnasium 1.0+ to register Atari ROMs
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack  # Imported SubprocVecEnv
from stable_baselines3.common.logger import configure  # Standard fresh-write logger for initial runs

def main():
    print("Checking hardware availability...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    print("Spawning 8 parallel Atari games across your Ryzen cores...")
    # Unlocked multi-threading by adding vec_env_cls=SubprocVecEnv
    env = make_atari_env(
        "ALE/Pong-v5", 
        n_envs=8, 
        seed=42, 
        env_kwargs={"render_mode": None},
        vec_env_cls=SubprocVecEnv
    )
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

    # --- THE LOG SPLITTER ENGINE (FRESH WRITE) ---
    # For a completely brand-new training run, standard write mode is fine
    log_dir = "./training_logs/"
    os.makedirs(log_dir, exist_ok=True)
    sb3_logger = configure(log_dir, ["stdout", "log"])
    model.set_logger(sb3_logger)
    # -------------------------------

    training_steps = 500_000  # Bumped up so you can push past the initial learning curve faster
    print(f"Starting multi-core training loop for {training_steps} total timesteps...")
    
    try:
        model.learn(total_timesteps=training_steps)
        print("Training complete!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")

    model_path = "ppo_pong_model"
    model.save(model_path)
    print(f"Model saved cleanly to {model_path}.zip")

if __name__ == "__main__":
    # Crucial protection macro for Windows multi-processing
    main()