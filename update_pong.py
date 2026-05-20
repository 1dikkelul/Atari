import os
import time
import torch
import ale_py  # Crucial for Gymnasium 1.0+ to register Atari ROMs
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.logger import configure  # Essential for file logging

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # 1. Re-create the identical environment setup
    print("Setting up environments...")
    env = make_atari_env("ALE/Pong-v5", n_envs=8, seed=42, env_kwargs={"render_mode": None}, vec_env_cls=SubprocVecEnv)
    env = VecFrameStack(env, n_stack=4)

    # 2. LOAD the existing saved model and OVERRIDE parameters
    print("Loading existing model weights with updated training dimensions...")
    
    # We explicitly inject our new, larger step and batch sizes right here
    model = PPO.load(
        "ppo_pong_model", 
        env=env, 
        device=device,
        n_steps=2048,     # <--- Changed from 128
        batch_size=512    # <--- Increased to keep optimization stable
    )

    # --- THE BREAKOUT INJECTION ---
    # Force the learning rate back to its default starting speed
    model.learning_rate = 0.00005    # Start with 0.00025 but then move it to 0.00005 for more final learning
    
    # Force it to value exploration again (default is usually 0.0)
    # This prevents the entropy from staying locked down at -0.105
    model.ent_coef = 0.0          #Started with 0.01 to force it to explore, but closer to endgame leaving it back to 0.0

    # --- THE APPEND-MODE LOG SPLITTER ENGINE ---
    import sys
    from stable_baselines3.common.logger import Logger, HumanOutputFormat

    log_dir = "./training_logs/"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "log.txt")

    # 1. Manually build the outputs. We open the file handler in 'a' (append) mode.
    stdout_format = HumanOutputFormat(sys.stdout)
    file_format = HumanOutputFormat(open(log_file_path, "a", encoding="utf-8"))

    # 2. Bind them into a custom SB3 Logger instance and inject it into the model
    sb3_logger = Logger(folder=log_dir, output_formats=[stdout_format, file_format])
    model.set_logger(sb3_logger)
    # -------------------------------------------

    # 3. Resume training for a longer stretch
    extra_steps = 10_000_000  # Bumped this up so you give the AI a real chance to learn!
    print(f"Resuming training for an additional {extra_steps} timesteps...")
    
    start_time = time.time()
    try:
        # ADDED: log_interval=100 stops it from writing a massive table every 16k steps.
        # It will now wait 100 updates before printing a status report.
        model.learn(
            total_timesteps=extra_steps, 
            reset_num_timesteps=False,
            log_interval=1  # <--- CONTROL LOGGING FREQUENCY HERE
        )
        print("Additional training complete!")
    except KeyboardInterrupt:
        print("\nTraining paused by user.")
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n--- Training Session Summary ---")
        print(f"Real-time elapsed: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
        print(f"--------------------------------\n")

    # 4. Overwrite model
    model.save("ppo_pong_model")
    print("Updated model saved as ppo_pong_model.zip")

if __name__ == "__main__":
    main()