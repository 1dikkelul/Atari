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
    
    # FIX: Switch to NoFrameskip-v4 to eliminate the double frame-skip bottleneck!
    raw_env_fn = lambda: AtariWrapper(gym.make("PongNoFrameskip-v4", render_mode="human"), screen_size=84)
    env = DummyVecEnv([raw_env_fn])
    
    # Stack 4 frames so the model can perceive ball velocity accurately
    env = VecFrameStack(env, n_stack=4)

    print("Loading your trained model...")
    # Point directly to your master weight archive name
    model = PPO.load("ppo_pong_model_v1", env=env, device=device)

    print("Starting match! Enjoy the show.")
    obs = env.reset()
    done = False
    
    try:
        while not done:
            # The model reads the clean (4, 84, 84) grayscale stack matching its training dimension
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