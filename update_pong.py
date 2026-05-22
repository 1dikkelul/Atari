import os
import time
import torch
import ale_py  # Crucial for Gymnasium 1.0+ to register Atari ROMs
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback

# --- LEARNING RATE DECAY FUNCTION ---
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# --- LIVE WEEKEND TIMEOUT CALLBACK ---
class WeekendTimeoutCallback(BaseCallback):
    """
    Stops training automatically if the elapsed wall-clock time 
    exceeds a pre-allocated max duration.
    """
    def __init__(self, max_hours: float, verbose=1):
        super(WeekendTimeoutCallback, self).__init__(verbose)
        self.max_seconds = max_hours * 3600
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            if self.verbose > 0:
                print(f"\n[TIMEOUT] Weekend timer limit reached ({elapsed/3600:.2f} hours). Stopping training loop safely...")
            return False  # Returning False completely breaks the PPO execution loop cleanly
        return True

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    print("Setting up training environments...")
    env = make_atari_env("ALE/Pong-v5", n_envs=8, seed=42, env_kwargs={"render_mode": None}, vec_env_cls=SubprocVecEnv)
    env = VecFrameStack(env, n_stack=4)

    print("Setting up evaluation environments...")
    eval_env = make_atari_env("ALE/Pong-v5", n_envs=4, seed=100, env_kwargs={"render_mode": None}, vec_env_cls=SubprocVecEnv)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    # --- UPDATED TO WORK ON VERSION 3 CHECKPOINTS ---
    model_name = "ppo_pong_model_v3"
    print(f"Loading existing model weights from {model_name}.zip...")
    
    model = PPO.load(
        model_name, 
        env=env, 
        device=device,
        n_steps=128,      
        batch_size=256    
    )

    # --- BREAKOUT INJECTION PARAMETERS ---
    model.clip_range = lambda progress_remaining: 0.2
    model.ent_coef = 0.01
    model.learning_rate = linear_schedule(2.5e-4)

    # --- LOG ENGINE SETUP ---
    import sys
    from stable_baselines3.common.logger import Logger, HumanOutputFormat

    log_dir = "./training_logs/"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "log.txt")

    stdout_format = HumanOutputFormat(sys.stdout)
    file_format = HumanOutputFormat(open(log_file_path, "a", encoding="utf-8"))
    sb3_logger = Logger(folder=log_dir, output_formats=[stdout_format, file_format])
    model.set_logger(sb3_logger)

    # --- CALLBACK COUPLING ---
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=20.0, verbose=1)
    eval_callback = EvalCallback(
        eval_env, 
        callback_on_new_best=callback_on_best, 
        eval_freq=max(2500, 20000 // 8), 
        log_path=log_dir, 
        best_model_save_path="./best_model/",
        deterministic=True, 
        verbose=1
    )

    # Set to 6.0 Hours max runtime limit
    timeout_callback = WeekendTimeoutCallback(max_hours=6.0, verbose=1)
    
    # Bundle both callbacks together into a single execution chain
    callbacks = [eval_callback, timeout_callback]

    extra_steps = 15_000_000  
    print(f"Resuming training for up to {extra_steps} timesteps with a 6.0 hour time ceiling...")
    
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=extra_steps, 
            reset_num_timesteps=False,   
            callback=callbacks,
            log_interval=1
        )
        print("\nTraining session ended successfully!")
    except KeyboardInterrupt:
        print("\nTraining paused by user via KeyboardInterrupt.")
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n--- Training Session Summary ---")
        print(f"Real-time elapsed: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
        print(f"--------------------------------\n")

    # Overwrites your v3 file safely
    model.save(model_name)
    print(f"Updated model saved cleanly back to {model_name}.zip")

if __name__ == "__main__":
    main()