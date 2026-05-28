import os
import glob
import pandas as pd

def scan_training_runs():
    search_path = "./training_runs/*/telemetry/progress.csv"
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        print("❌ No training telemetry sheets found under './training_runs/*/telemetry/progress.csv'")
        print("Ensure your training folders are structured inside the './training_runs/' directory.")
        return

    print("\n" + "=" * 125)
    print(f"{'Experiment Run':<18} | {'Total Steps':<12} | {'Final Rollout Reward':<22} | {'Best Eval Reward':<16} | {'Peak FPS':<8} | {'Duration':<9} | {'ExplVar':<7} | {'ApproxKL'}")
    print("=" * 125)

    for csv_path in sorted(csv_files):
        # Extract the version tag name from the folder path structure
        path_parts = csv_path.split(os.sep)
        version_tag = path_parts[-3] if len(path_parts) >= 3 else "Unknown"

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
                
            # 1. Total timesteps is the maximum value reached over the entire run
            steps = df["time/total_timesteps"].max() if "time/total_timesteps" in df.columns else 0
            steps_str = f"{int(steps):,}" if pd.notna(steps) else "N/A"
            
            # 2. Extract last valid rollout mean reward (skips empty evaluation trigger rows)
            rollout_series = df["rollout/ep_rew_mean"].dropna() if "rollout/ep_rew_mean" in df.columns else pd.Series()
            rollout_rew = rollout_series.iloc[-1] if not rollout_series.empty else None
            rollout_str = f"{rollout_rew:+6.2f}" if rollout_rew is not None else f"{'N/A':<7}"
            
            # 3. Peak evaluation callback score recorded across the entire lifespan
            eval_col = "eval/mean_reward"
            if eval_col in df.columns and df[eval_col].notna().any():
                best_eval = df[eval_col].max()
                eval_str = f"{best_eval:+6.2f}"
            else:
                eval_str = f"{'N/A':<6}"

            # 4. Extract the last valid tracking speed throughput
            fps_series = df["time/fps"].dropna() if "time/fps" in df.columns else pd.Series()
            fps = int(fps_series.iloc[-1]) if not fps_series.empty else 0
            
            # 5. Maximum elapsed wall-clock session duration
            elapsed_seconds = df["time/time_elapsed"].max() if "time/time_elapsed" in df.columns else 0
            duration_str = f"{elapsed_seconds / 3600:.2f}h" if pd.notna(elapsed_seconds) else "N/A"
            
            # 6. Extract latest valid training optimization vectors
            ev_series = df["train/explained_variance"].dropna() if "train/explained_variance" in df.columns else pd.Series()
            expl_var = ev_series.iloc[-1] if not ev_series.empty else None
            ev_str = f"{expl_var:.3f}" if expl_var is not None else "N/A"
            
            kl_series = df["train/approx_kl"].dropna() if "train/approx_kl" in df.columns else pd.Series()
            approx_kl = kl_series.iloc[-1] if not kl_series.empty else None
            kl_str = f"{approx_kl:.5f}" if approx_kl is not None else "N/A"

            print(f"{version_tag:<18} | "
                  f"{steps_str:<12} | "
                  f"{rollout_str:<22} | "
                  f"{eval_str:<16} | "
                  f"{fps:<8} | "
                  f"{duration_str:<9} | "
                  f"{ev_str:<7} | "
                  f"{kl_str}")
                  
        except Exception as e:
            print(f"{version_tag:<18} | Failed to read telemetry sheet. Error: {str(e)}")

    print("=" * 125 + "\n")

if __name__ == "__main__":
    scan_training_runs()