import os
import re
import sys
import time
import torch
import numpy as np
from gymnasium import spaces
from pettingzoo.atari import pong_v3
import supersuit as ss
from agilerl.utils.utils import create_population
from stable_baselines3 import PPO

# --- EXACT PATH CONFIGURATIONS ---
EVO_DIR = "./training_runs/evolutionary_league"
SINGLE_AGENT_PATH = "./training_runs/single_agent/ppo_pong_model.zip"
PLAYBACK_DELAY_SEC = 0.04


class ChannelFirstParallelWrapper:
    def __init__(self, env):
        self.env = env
        self.possible_agents = env.possible_agents

    @property
    def agents(self):
        return self.env.agents

    def _transpose_obs(self, observations):
        return {
            agent: np.transpose(obs, (2, 0, 1))
            for agent, obs in observations.items()
        }

    def observation_space(self, agent):
        space = self.env.observation_space(agent)
        if len(space.shape) != 3:
            return space

        channels = space.shape[2]
        height = space.shape[0]
        width = space.shape[1]
        return spaces.Box(
            low=space.low.min(),
            high=space.high.max(),
            shape=(channels, height, width),
            dtype=space.dtype,
        )

    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self, *args, **kwargs):
        observations, infos = self.env.reset(*args, **kwargs)
        return self._transpose_obs(observations), infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return self._transpose_obs(observations), rewards, terminations, truncations, infos

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


def _get_latest_frame(obs):
    arr = np.array(obs)
    if arr.ndim == 3 and arr.shape[0] >= 1:
        return arr[-1]
    return arr


def _estimate_ball_y(frame):
    h, w = frame.shape
    center = frame[:, w // 4 : (3 * w) // 4]
    ys, _ = np.where(center > 0)
    return int(np.median(ys)) if ys.size > 0 else h // 2


def _estimate_paddle_y(frame, side):
    h, w = frame.shape
    strip = frame[:, :12] if side == "left" else frame[:, w - 12 :]
    ys, _ = np.where(strip > 0)
    return int(np.median(ys)) if ys.size > 0 else h // 2


def heuristic_pong_action(obs, side):
    frame = _get_latest_frame(obs)
    ball_y = _estimate_ball_y(frame)
    paddle_y = _estimate_paddle_y(frame, side)

    if ball_y < paddle_y - 1:
        return 3
    if ball_y > paddle_y + 1:
        return 2
    return 0


def load_sb3_model_compat(model_path, observation_space=None, action_space=None):
    # Some older SB3 checkpoints were pickled with numpy._core.* module paths.
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np.core
    if "numpy._core.numeric" not in sys.modules:
        sys.modules["numpy._core.numeric"] = np.core.numeric

    # Ignore stale training/runtime state in old checkpoints.
    custom_objects = {
        "_last_obs": None,
        "_last_episode_starts": None,
        "_last_original_obs": None,
        "ep_info_buffer": None,
        "ep_success_buffer": None,
    }
    if observation_space is not None:
        custom_objects["observation_space"] = observation_space
    if action_space is not None:
        custom_objects["action_space"] = action_space

    return PPO.load(model_path, custom_objects=custom_objects)

def get_available_generations():
    """Scans evolutionary_league and returns sorted pairs of (gen_num, path)."""
    if not os.path.exists(EVO_DIR):
        return []
    
    files = os.listdir(EVO_DIR)
    gen_models = []
    for f in files:
        if f.startswith("pong_champ_gen_") and f.endswith(".pt"):
            match = re.search(r"pong_champ_gen_(\d+)\.pt", f)
            if match:
                gen_num = int(match.group(1))
                gen_models.append((gen_num, os.path.join(EVO_DIR, f)))
                
    return sorted(gen_models, key=lambda x: x[0])

def select_model_interactive():
    """Interactive menu to choose which AgileRL generation to load."""
    models = get_available_generations()
    if not models:
        print(f"❌ No evolutionary models found in: {EVO_DIR}")
        return None
        
    print("\n--- 🧬 CHOOSE YOUR AGILERL LEAGUE CHAMPION ---")
    for gen, path in models:
        print(f" [{gen}] Generation {gen} Checkpoint ({os.path.basename(path)})")
        
    while True:
        try:
            choice = int(input("\nEnter Generation Number to deploy (e.g., 100): "))
            selected = [m for m in models if m[0] == choice]
            if selected:
                return selected[0][1]
            print("Invalid generation number. Pick from the list above.")
        except ValueError:
            print("Please enter a valid integer.")

def main():
    print("========================================================")
    print("---          ATARI PONG TRI-REALM ARENA              ---")
    print("========================================================")
    
    # 1. Choose the AgileRL League Model (Left Paddle)
    evo_model_path = select_model_interactive()
    if not evo_model_path:
        return
        
    # 2. Choose the Opponent Style (Right Paddle)
    print("\n--- ⚔️ SELECT OPPONENT STYLE FOR RIGHT PADDLE ---")
    print(" [1] Heuristic Atari-Style AI Opponent")
    print(" [2] Manual Keyboard Controls (Play Yourself via W/S or Arrow Keys)")
    print(" [3] Stable-Baselines3 Single-Agent Model (ppo_pong_model.zip)")
    
    opponent_mode = input("\nEnter choice (1, 2, or 3): ").strip()
    
    # Validate SB3 model existence if Mode 3 selected
    sb3_model = None
    sb3_requested = opponent_mode == "3"
    if sb3_requested and not os.path.exists(SINGLE_AGENT_PATH):
        print(f"❌ Cannot find Stable-Baselines3 model at: {SINGLE_AGENT_PATH}")
        return

    # 3. Spin up native PettingZoo with Visual Rendering enabled
    print("\nInitializing native Atari ROM runtime environment...")
    env = pong_v3.parallel_env(obs_type='rgb_image', render_mode='human', max_cycles=100000)
    
    # Identical SuperSuit transformations matching your training pipeline
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.color_reduction_v0(env, mode='full')
    env = ss.resize_v1(env, 84, 84)
    env = ss.frame_stack_v2(env, 4)
    
    # Match training pipeline exactly by transposing HWC -> CHW, rather than reshaping memory layout.
    env = ChannelFirstParallelWrapper(env)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agents = env.possible_agents  # ['first_0', 'second_0']
    # In this env variant, second_0 renders as the left/orange paddle.
    model_agent = "second_0"
    opponent_agent = "first_0"

    if sb3_requested:
        print(f"📥 Loading Stable-Baselines3 model into memory...")
        sb3_model = load_sb3_model_compat(
            SINGLE_AGENT_PATH,
            observation_space=env.observation_space(opponent_agent),
            action_space=env.action_space(opponent_agent),
        )
    
    # 4. Instantiate a dummy AgileRL container to hold the state dictionary weights
    NET_CONFIG = {
        "encoder_config": {
            "channel_size": [32, 64, 64],
            "kernel_size": [8, 4, 3],
            "stride_size": [4, 2, 1]
        }
    }
    INIT_HP = {
        "ALGO": "PPO", "BATCH_SIZE": 256, "LR": 2.5e-4, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
        "SHARE_ENCODERS": False, "ACTION_MASKED": False
    }
    
    dummy_pop = create_population(
        algo=INIT_HP["ALGO"],
        observation_space=env.observation_space(agents[0]),
        action_space=env.action_space(agents[0]),
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        population_size=1,
        device=device
    )
    league_brain = dummy_pop[0]
    
    print(f"Injecting AgileRL structural weights from: {os.path.basename(evo_model_path)}")
    league_brain.load_checkpoint(evo_model_path)
    
    print("\n🚀 Arena Active! Click on the game window. Press 'ESC' to exit cleanly.")
    
    states, infos = env.reset()
    running = True
    try:
        while env.agents and running:
            actions = {}

            # --- LEFT (ORANGE): AgileRL League Brain ---
            if model_agent in env.agents:
                state_arr = np.array(states[model_agent])
                state_tensor = torch.from_numpy(state_arr).unsqueeze(0).to(device)

                # Match AgileRL PPO API used in training code.
                action, _, _, _ = league_brain.get_action(state_tensor)
                actions[model_agent] = int(np.array(action).squeeze())

            # --- RIGHT (GREEN): Selected Opponent ---
            if opponent_agent in env.agents:
                if opponent_mode == "1":
                    actions[opponent_agent] = heuristic_pong_action(states[opponent_agent], side="right")

                elif opponent_mode == "2":
                    import pygame

                    pygame.event.pump()
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_UP] or keys[pygame.K_w]:
                        actions[opponent_agent] = 2
                    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                        actions[opponent_agent] = 3
                    else:
                        actions[opponent_agent] = 0

                    if keys[pygame.K_ESCAPE]:
                        running = False

                elif opponent_mode == "3":
                    sb3_input = np.expand_dims(np.array(states[opponent_agent]), axis=0)
                    sb3_action, _ = sb3_model.predict(sb3_input, deterministic=True)
                    actions[opponent_agent] = int(sb3_action[0])

            # Only poll pygame events when manual keyboard controls are active.
            if opponent_mode == "2":
                import pygame

                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False

            # Step the underlying emulation engine forward
            next_states, rewards, terminations, truncations, infos = env.step(actions)
            states = next_states

            # Keep playback watchable for humans instead of running at emulator max speed.
            time.sleep(PLAYBACK_DELAY_SEC)

    except KeyboardInterrupt:
        print("\nArena context closed out.")
    finally:
        env.close()
        print("\n========================================================")
        print("Arena Exhibition Completed Cleanly.")
        print("========================================================\n")

if __name__ == "__main__":
    main()