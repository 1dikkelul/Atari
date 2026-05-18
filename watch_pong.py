import time
import torch
import ale_py  # Essential for registering Atari ROMs
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def main():
    print("Launching Full-Resolution Atari Pong window...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Correct Order: Create raw game -> Wrap with AtariWrapper -> Vectorize with DummyVecEnv
    raw_env_fn = lambda: AtariWrapper(gym.make("ALE/Pong-v5", render_mode="human"), screen_size=84)
    env = DummyVecEnv([raw_env_fn])
    
    # Stack 4 frames so the model can perceive ball velocity
    env = VecFrameStack(env, n_stack=4)

    print("Loading your trained model...")
    model = PPO.load("ppo_pong_model", env=env, device=device)

    print("Starting match! Enjoy the show.")
    obs = env.reset()
    done = False
    
    try:
        while not done:
            # The model reads the hidden (4, 84, 84) grayscale stack
            action, _states = model.predict(obs, deterministic=True)
            
            # The environment steps forward, updating your full-color window natively
            obs, rewards, dones, infos = env.step(action)
            
            # Match normal human gameplay speed
            time.sleep(0.01)
            
            if dones[0]:
                print("Game Over! The match has concluded.")
                done = True
                
    except KeyboardInterrupt:
        print("\nVisual playback stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()