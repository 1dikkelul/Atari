import os
import sys
import time
import torch
import ale_py  # Crucial for Gymnasium to register Atari ROMs
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback

# --- ADVANCED TERM PROGRESS MONITOR ---
class ConsolidatedProgressLogger(BaseCallback):
    """
    Tracks rollout milestones, printing a clean, single-line telemetry status
    with active performance throughput metrics and real-world ETA tracking.
    """
    def __init__(self, total_steps: int, verbose=1):
        super(ConsolidatedProgressLogger, self).__init__(verbose)
        self.total_steps = total_steps
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # We hook into the end of a full collection rollout block to update the screen
        if self.num_timesteps % (self.model.n_steps * self.model.n_envs) == 0:
            elapsed_time = time.time() - self.start_time
            fps = int(self.num_timesteps / elapsed_time) if elapsed_time > 0 else 0
            
            # Progress percentages and ETA calculations
            progress_pct = (self.num_timesteps / self.total_steps) * 100
            remaining_steps = self.total_steps - self.num_timesteps
            eta_seconds = remaining_steps / fps if fps > 0 else 0
            
            print(
                f"🚀 [PROGRESS] Steps: {self.num_timesteps:,}/{self.total_steps:,} "
                f"({progress_pct:.1f}%) | "
                f"Speed: {fps} FPS | "
                f"Elapsed: {elapsed_time/3600:.2f}h | "
                f"ETA: {eta_seconds/3600:.2f}h"
            )
        return True

# --- SAFE WEEKEND TIMEOUT CALLBACK ---
class WeekendTimeoutCallback(BaseCallback):
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
                print(f"\n[TIMEOUT] Timer limit reached ({elapsed/3600:.2f} hours). Stopping loop safely...")
            return False  
        return True

def main():
    print("Checking hardware availability...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # =========================================================================
    # CONFIGURATION & HIERARCHICAL PATH CLEANUP
    # =========================================================================
    VERSION_TAG = "v6_temp"
    
    # Everything is anchored inside a single master folder path
    MASTER_RUNS_DIR = "./training_runs"
    version_dir = os.path.join(MASTER_RUNS_DIR, VERSION_TAG)
    
    # Structural subdirectories mapping the entire lifecycle of this specific run
    log_dir = os.path.join(version_dir, "telemetry")
    best_model_dir = os.path.join(version_dir, "best_checkpoint")
    output_model_path = os.path.join(version_dir, f"ppo_pong_model_{VERSION_TAG}")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    print(f"\n📁 Experiment Root Established: {version_dir}")
    print(f"   ---> Telemetry Destination: {log_dir}")
    print(f"   ---> Evaluation Target:     {best_model_dir}\n")

    print("Creating training and evaluation environments...")
    # Clean 4-frame skip baseline matching our wrapper physics verification
    env = make_atari_env(
        "PongNoFrameskip-v4", 
        n_envs=16, 
        seed=42, 
        vec_env_cls=SubprocVecEnv
    )
    env = VecFrameStack(env, n_stack=4)

    eval_env = make_atari_env(
        "PongNoFrameskip-v4", 
        n_envs=4, 
        seed=100, 
        vec_env_cls=SubprocVecEnv
    )
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    print("Building CNN Model with flat learning rate capabilities...")
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        learning_rate=2.5e-4,                   
        n_steps=128,                            # 128 steps * 16 envs = 2048 token batch size
        batch_size=256,
        n_epochs=4,
        ent_coef=0.01,                          
        vf_coef=0.5,
        clip_range=0.2,                         
        device=device
    )

    sb3_logger = configure(log_dir, ["stdout", "log", "csv"])
    model.set_logger(sb3_logger)

    target_steps = 20_000_000 

    # =========================================================================
    # CALLBACK COUPLING
    # =========================================================================
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=20.0, verbose=1)
    
    eval_callback = EvalCallback(
        eval_env, 
        callback_on_new_best=callback_on_best, 
        eval_freq=max(2500, 40000 // 16), 
        log_path=log_dir, 
        best_model_save_path=best_model_dir,
        deterministic=True, 
        verbose=1
    )

    timeout_callback = WeekendTimeoutCallback(max_hours=12.0, verbose=1)
    progress_callback = ConsolidatedProgressLogger(total_steps=target_steps, verbose=1)
    
    # Group our architectural control modules together
    callbacks = [eval_callback, timeout_callback, progress_callback]

    print(f"Starting final master run up to {target_steps:,} steps...")
    start_wall_time = time.time()
    
    try:
        model.learn(
            total_timesteps=target_steps, 
            callback=callbacks,
            log_interval=10  # Reduced to 10 to let the custom progress logger handle granular tracking
        )
        print("\nTarget score reached or loop finished! Proceeding to save...")
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving progress...")
    finally:
        end_wall_time = time.time()
        elapsed = end_wall_time - start_wall_time
        
        print(f"\n" + "="*60)
        print(f"--- RUN EXECUTION SUMMARY ({VERSION_TAG.upper()}) ---")
        print(f"="*60)
        print(f"Total Session Duration: {elapsed:.2f} seconds ({elapsed / 3600:.2f} hours)")
        print(f"------------------------------------------------------------")
        print(f"📂 Master Directory:   {version_dir}")
        print(f"📊 CSV Telemetry Log:  {os.path.join(log_dir, 'progress.csv')}")
        print(f"🏆 Best Eval Check:    {os.path.join(best_model_dir, 'best_model.zip')}")
        print(f"💾 Final Model Weight: {output_model_path}.zip")
        print(f"="*60 + "\n")
        
        # Save structural checkpoint safely inside its clean subdirectory folder
        model.save(output_model_path)
        print(f"Model saved cleanly as {output_model_path}.zip")

if __name__ == "__main__":
    main()