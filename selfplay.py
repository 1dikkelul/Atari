import os
import sys
import time
import torch
import numpy as np
from gymnasium import spaces

# Multi-Agent Environment Framework
from pettingzoo.atari import pong_v3
import supersuit as ss

# AgileRL Core Components
from agilerl.utils.utils import create_population
from agilerl.hpo.tournament import TournamentSelection as EvolutionarySearch

# --- ADVANCED MARL PROGRESS MONITOR ---
class AgileProgressMonitor:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.start_time = time.time()
        self.last_print_time = time.time()

    def check_and_print(self, current_steps: int, fitness_scores: list):
        current_time = time.time()
        # Output status dashboard every 60 seconds for optimal feedback loops
        if current_time - self.last_print_time >= 60.0:  
            elapsed_time = current_time - self.start_time
            fps = int(current_steps / elapsed_time) if elapsed_time > 0 else 0
            progress_pct = (current_steps / self.total_steps) * 100
            remaining_steps = self.total_steps - current_steps
            eta_hours = (remaining_steps / fps) / 3600 if fps > 0 else 0
            
            best_fit = max(fitness_scores) if fitness_scores else float('-inf')
            avg_fit = np.mean(fitness_scores) if fitness_scores else float('-inf')

            print(
                f"🧬 [EVO TELEMETRY] Steps: {current_steps:,}/{self.total_steps:,} ({progress_pct:.2f}%) | "
                f"Performance: {fps} FPS | "
                f"Best Generation Fit: {best_fit:.2f} | Avg Fit: {avg_fit:.2f} | "
                f"ETA: {eta_hours:.2f}h"
            )
            self.last_print_time = current_time


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

def make_agile_pong_env():
    """
    Spawns a true parallel multi-agent PettingZoo Pong environment.
    """
    env = pong_v3.parallel_env(
        obs_type='rgb_image', 
        max_cycles=100000
    )
    
    env = ss.max_observation_v0(env, 2)            
    env = ss.frame_skip_v0(env, 4)                 
    env = ss.color_reduction_v0(env, mode='full')  
    env = ss.resize_v1(env, 84, 84)                
    env = ss.frame_stack_v2(env, 4)                
    env = ChannelFirstParallelWrapper(env)
    return env

def main():
    print("Verifying hardware acceleration...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"AgileRL Evolution running on: {device.upper()}")

    VERSION_TAG = "agile_marl_v1"
    MASTER_RUNS_DIR = "./training_runs"
    version_dir = os.path.join(MASTER_RUNS_DIR, VERSION_TAG)
    checkpoint_dir = os.path.join(version_dir, "evolved_brains")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"📁 Saving evolutionary checkpoints cleanly to: {checkpoint_dir}\n")

    init_env = make_agile_pong_env()
    agents = init_env.possible_agents  
    
    state_dim = init_env.observation_space(agents[0]).shape
    action_dim = init_env.action_space(agents[0]).n

    NET_CONFIG = {
        "encoder_config": {
            "channel_size": [32, 64, 64],
            "kernel_size": [8, 4, 3],
            "stride_size": [4, 2, 1]
        }
    }

    INIT_HP = {
        "ALGO": "PPO",                     
        "BATCH_SIZE": 256,                 
        "LR": 2.5e-4,                      
        "GAMMA": 0.99,                     
        "GAE_LAMBDA": 0.95,                
        "SHARE_ENCODERS": False,
        "ACTION_MASKED": False
    }

    MUTATION_PARAMS = {
        "no_mut": 0.4,                     
        "architect_mut": 0.2,              
        "param_mut": 0.2,                  
        "lr_mut": 0.2,                     
        "mutation_strength": 0.1
    }

    POP_SIZE = 4  
    print(f"Spawning genetic population pool consisting of {POP_SIZE} active strains...")
    
    population = create_population(
        algo=INIT_HP["ALGO"],
        observation_space=init_env.observation_space(agents[0]), 
        action_space=init_env.action_space(agents[0]),           
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        population_size=POP_SIZE,
        device=device        
    )

    init_env.close()

    evolutionary_manager = EvolutionarySearch(
        tournament_size=2,                 
        elitism=True,                      
        population_size=POP_SIZE,          
        eval_loop=1                        
    )

    env = make_agile_pong_env()
    
    TOTAL_TARGET_STEPS = 10_000_000
    MAX_HOURS = 12.0
    EVO_STEP_INTERVAL = 100_000        # Every 100,000 steps triggers tournament updates
    
    global_step_counter = 0
    next_evo_milestone = EVO_STEP_INTERVAL
    evo_generation_count = 0
    
    # Track historical fitness scores for each strain to enable proper selection
    brain_fitness_trackers = {id(brain): [] for brain in population}
    
    progress_monitor = AgileProgressMonitor(total_steps=TOTAL_TARGET_STEPS)
    start_wall_clock = time.time()

    print("Beginning AgileRL generation loops...")
    try:
        while global_step_counter < TOTAL_TARGET_STEPS:
            if (time.time() - start_wall_clock) > (MAX_HOURS * 3600):
                print(f"\n[TIMEOUT] Reached run threshold limit of {MAX_HOURS} hours. Exiting safely...")
                break

            # --- STEP 1: EXPERIENCE COLLECTION (SELF-PLAY MATCHES) ---
            states, infos = env.reset()
            episode_rewards = {agent: 0 for agent in agents}
            episode_brains = dict(
                zip(
                    agents,
                    np.random.choice(population, size=len(agents), replace=False),
                )
            )
            episode_rollouts = {
                id(brain): {
                    "brain": brain,
                    "observations": [],
                    "actions": [],
                    "log_probs": [],
                    "rewards": [],
                    "dones": [],
                    "values": [],
                    "next_obs": None,
                    "next_done": None,
                }
                for brain in episode_brains.values()
            }
            
            while env.agents:
                actions = {}
                for agent_id in env.agents:
                    active_brain = episode_brains[agent_id]
                    
                    state_tensor = torch.from_numpy(np.array(states[agent_id])).unsqueeze(0).to(device)
                    action, log_prob, _, value = active_brain.get_action(state_tensor)

                    brain_rollout = episode_rollouts[id(active_brain)]
                    brain_rollout["observations"].append(np.array(states[agent_id]))
                    brain_rollout["actions"].append(np.array(action).squeeze())
                    brain_rollout["log_probs"].append(np.array(log_prob).squeeze())
                    brain_rollout["values"].append(np.array(value).squeeze())

                    actions[agent_id] = int(np.array(action).squeeze())

                next_states, rewards, terminations, truncations, infos = env.step(actions)
                
                for agent_id in actions.keys():
                    active_brain = episode_brains[agent_id]
                    brain_rollout = episode_rollouts[id(active_brain)]
                    done = terminations[agent_id] or truncations[agent_id]
                    brain_rollout["rewards"].append(np.array(rewards[agent_id], dtype=np.float32))
                    brain_rollout["dones"].append(np.array(done, dtype=np.float32))
                    brain_rollout["next_obs"] = np.array(next_states[agent_id])
                    brain_rollout["next_done"] = np.array(done, dtype=np.float32)
                    episode_rewards[agent_id] += rewards[agent_id]
                    global_step_counter += 1

                states = next_states
            
            # Record match results to update running fitness averages
            for agent_id, reward in episode_rewards.items():
                active_brain = episode_brains[agent_id]
                brain_fitness_trackers[id(active_brain)].append(reward)
            
            print(f"🎮 [MATCH END] Global Steps: {global_step_counter:,} | "
                  f"Left Paddle Score: {episode_rewards[agents[0]]:.1f} | "
                  f"Right Paddle Score: {episode_rewards[agents[1]]:.1f}")

            # --- STEP 2: BRAIN OPTIMIZATION GRADIENTS ---
            for rollout in episode_rollouts.values():
                if rollout["rewards"]:
                    experiences = (
                        rollout["observations"],
                        rollout["actions"],
                        rollout["log_probs"],
                        rollout["rewards"],
                        rollout["dones"],
                        rollout["values"],
                        rollout["next_obs"],
                        rollout["next_done"],
                    )
                    rollout["brain"].learn(experiences)

            # --- STEP 3: THE GENETIC TOURNAMENT ELIMINATION ---
            # FIX: Use a directional threshold check so steps can never leap past checkpoints
            if global_step_counter >= next_evo_milestone:
                evo_generation_count += 1
                print(f"\n🏆 [TOURNAMENT] Generation {evo_generation_count} evaluation triggered at {global_step_counter:,} steps!")
                
                # FIX: Compute the explicit mean performance scores and manually bind them to the networks
                for brain in population:
                    history = brain_fitness_trackers[id(brain)]
                    mean_fitness = np.mean(history) if history else -21.0
                    brain.fitness.append(mean_fitness)
                    # Clear training history tracking buffer for the upcoming generation cycle
                    brain_fitness_trackers[id(brain)] = []
                
                # Extract clean arrays for performance dashboard output logs
                fitness_scores = [brain.fitness[-1] for brain in population]
                
                # Display structural details about your neural networks' hyperparameters
                print("📋 Current Population Metrics:")
                for idx, brain in enumerate(population):
                    current_lr = brain.lr if hasattr(brain, 'lr') else INIT_HP["LR"]
                    print(f"  🔹 Strain {idx+1} | Mean Match Reward: {brain.fitness[-1]:.2f} | Learning Rate: {current_lr}")
                
                # Execute evolutionary selection via tournament architecture
                elite_agent, population = evolutionary_manager.select(population)
                
                # Re-align tracking mapping buffers for newly generated mutations or clones
                old_trackers = brain_fitness_trackers.copy()
                brain_fitness_trackers = {id(brain): old_trackers.get(id(brain), []) for brain in population}
                
                # Save the reigning champion model cleanly to disk
                
                # Safe array guard in case a generation experiences completely identical scores
                champion_idx = int(np.argmax(fitness_scores)) if len(fitness_scores) > 0 else 0
                save_path = os.path.join(checkpoint_dir, f"pong_champ_gen_{evo_generation_count}.pt")

                # Modernized snake_case API method call
                population[champion_idx].save_checkpoint(save_path)

                print(f"💾 Saved elite genetic snapshot cleanly to: {save_path}\n")
                
                # Target the next sequential milestone interval bound safely
                next_evo_milestone += EVO_STEP_INTERVAL

            # Run regular 60-second console dashboard telemetry updates
            current_generation_fitness = [np.mean(brain_fitness_trackers[id(b)]) if brain_fitness_trackers[id(b)] else 0.0 for b in population]
            progress_monitor.check_and_print(global_step_counter, current_generation_fitness)

    except KeyboardInterrupt:
        print("\nExecution broken manually by keyboard. Wrapping up files safely...")
    finally:
        env.close()
        print(f"\n========================================================")
        print(f"--- AGILE_MARL RUN COMPLETE ---")
        print(f"Total Combined Step Counts Achieved: {global_step_counter:,}")
        print(f"Target Checkpoint Location: {checkpoint_dir}")
        print(f"========================================================\n")

if __name__ == "__main__":
    main()