import os
import sys
import torch
import ale_py  # Crucial for Gymnasium 1.0+ to register Atari ROMs
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

def main():
    print("Checking hardware availability...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    print("Creating training and evaluation environments...")
    # Training Environment Setup
    env = make_atari_env(
        "ALE/Pong-v5", 
        n_envs=8, 
        seed=42, 
        env_kwargs={"render_mode": None},
        vec_env_cls=SubprocVecEnv
    )
    env = VecFrameStack(env, n_stack=4)
    # Note: PPO wraps 'env' in VecTransposeImage automatically under the hood

    # Evaluation Environment Setup
    eval_env = make_atari_env(
        "ALE/Pong-v5", 
        n_envs=4, 
        seed=100, 
        env_kwargs={"render_mode": None},
        vec_env_cls=SubprocVecEnv
    )
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)  # Fixes the channel order warning for PyTorch

    # --- TUNING HYPERPARAMETERS FOR EXPLORATION & PONG ---
    print("Building the Core CNN Model with updated exploration parameters...")
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        learning_rate=2.5e-4,     # Standard stable LR for Atari
        n_steps=128,              # 128 steps per env = 1024 total steps per iteration
        batch_size=256,
        n_epochs=4,
        ent_coef=0.01,            # Forces the agent to keep exploring alternative moves
        vf_coef=0.5,              # Keeps value loss scaling in check
        clip_range=0.1,           # Tighter clipping for more stable policy updates on pixels
        device=device
    )

    log_dir = "./training_logs/"
    os.makedirs(log_dir, exist_ok=True)
    sb3_logger = configure(log_dir, ["stdout", "log"])
    model.set_logger(sb3_logger)

    # --- AUTOMATED STOPPING & EVAL ENGINE ---
    # Stops training completely once the evaluation environment reaches an average score of 20.0
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=20.0, verbose=1)
    
    # Evaluates the model every 20,000 steps using clean environments
    eval_callback = EvalCallback(
        eval_env, 
        callback_on_new_best=callback_on_best, 
        eval_freq=max(2500, 20000 // 8), 
        log_path=log_dir, 
        best_model_save_path="./best_model/",
        deterministic=True, 
        verbose=1
    )

    # Set to an incredibly massive number so it practically runs forever until stopped by code or user
    infinite_steps = 100_000_000 
    print(f"Starting endless training loop. Will auto-stop at an evaluation score of 20.0...")
    
    model_path = "ppo_pong_model_v2"
    try:
        model.learn(total_timesteps=infinite_steps, callback=eval_callback)
        print("\nTarget score reached or loop finished! Saving final model...")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user via KeyboardInterrupt. Saving current progress...")

    model.save(model_path)
    print(f"Model saved cleanly to {model_path}.zip")

if __name__ == "__main__":
    main()