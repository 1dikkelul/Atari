import time
from pathlib import Path
import sys
import numpy as np
import torch
import ale_py  # Essential for registering Atari ROMs
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage


def resolve_sb3_model_path(model_name: str) -> str:
    model_path = Path(model_name)

    # SB3 expects the .zip archive path when loading from disk.
    if model_path.is_dir():
        zipped = model_path.with_suffix(".zip")
        if zipped.exists():
            return str(zipped)

    if model_path.suffix != ".zip":
        zipped = model_path.with_suffix(".zip")
        if zipped.exists():
            return str(zipped)

    if model_path.exists() and model_path.is_file():
        return str(model_path)

    raise FileNotFoundError(
        f"Could not find SB3 model file for '{model_name}'. "
        f"Tried '{model_path}' and '{model_path.with_suffix('.zip')}'."
    )


def load_sb3_model_compat(model_path: str, device: str, observation_space, action_space):
    # Older SB3 checkpoints may reference legacy numpy._core module paths.
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np.core
    if "numpy._core.numeric" not in sys.modules:
        sys.modules["numpy._core.numeric"] = np.core.numeric

    # Some old pickles pass a BitGenerator class instead of its name string.
    # Newer NumPy expects a string and raises ValueError without this shim.
    try:
        import numpy.random._pickle as np_pickle

        if not hasattr(np_pickle, "_original_bit_generator_ctor"):
            np_pickle._original_bit_generator_ctor = np_pickle.__bit_generator_ctor

            def _compat_bit_generator_ctor(bit_generator_name="MT19937"):
                if isinstance(bit_generator_name, type):
                    bit_generator_name = bit_generator_name.__name__
                return np_pickle._original_bit_generator_ctor(bit_generator_name)

            np_pickle.__bit_generator_ctor = _compat_bit_generator_ctor
    except Exception:
        pass

    # Ignore stale runtime state serialized in older checkpoints.
    custom_objects = {
        "_last_obs": None,
        "_last_episode_starts": None,
        "_last_original_obs": None,
        "ep_info_buffer": None,
        "ep_success_buffer": None,
        "observation_space": observation_space,
        "action_space": action_space,
    }
    return PPO.load(model_path, device=device, custom_objects=custom_objects)

def main():
    print("Launching Full-Resolution Atari Pong window...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # FIX: Switch to NoFrameskip-v4 to eliminate the double frame-skip bottleneck!
    raw_env_fn = lambda: AtariWrapper(gym.make("PongNoFrameskip-v4", render_mode="human"), screen_size=84)
    env = DummyVecEnv([raw_env_fn])
    
    # Stack 4 frames so the model can perceive ball velocity accurately
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    print("Loading your trained model...")
    model_path = resolve_sb3_model_path("ppo_pong_model_v1")
    model = load_sb3_model_compat(
        model_path,
        device=device,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

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