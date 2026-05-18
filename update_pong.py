import torch
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # 1. Re-create the identical environment setup
    print("Setting up environments...")
    env = make_atari_env("ALE/Pong-v5", n_envs=8, seed=42, env_kwargs={"render_mode": None})
    env = VecFrameStack(env, n_stack=4)

    # 2. LOAD the existing saved model instead of making a new one
    print("Loading existing model weights...")
    model = PPO.load("ppo_pong_model", env=env, device=device)

    # 3. Resume training for a longer stretch
    extra_steps = 500_000
    print(f"Resuming training for an additional {extra_steps} timesteps...")
    
    try:
        model.learn(total_timesteps=extra_steps, reset_num_timesteps=False)
        print("Additional training complete!")
    except KeyboardInterrupt:
        print("\nTraining paused by user.")

    # 4. Overwrite or save as a new version
    model.save("ppo_pong_model_v2")
    print("Updated model saved cleanly as ppo_pong_model_v2.zip")

if __name__ == "__main__":
    main()