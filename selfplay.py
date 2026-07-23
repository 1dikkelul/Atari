import os
import sys
import time
import json
import torch
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

# Multi-Agent Environment Framework
from pettingzoo.atari import pong_v3
import supersuit as ss

# AgileRL Core Components
from agilerl.utils.utils import create_population
from agilerl.hpo.tournament import TournamentSelection as EvolutionarySearch
from agilerl.hpo.mutation import Mutations

# --- ADVANCED MARL PROGRESS MONITOR ---
class AgileProgressMonitor:
    def __init__(self, total_steps: int, frame_skip: int, num_agents: int):
        self.total_steps = total_steps
        self.frame_skip = frame_skip
        self.num_agents = num_agents
        self.start_time = time.time()
        self.last_print_time = time.time()

    def check_and_print(self, current_steps: int, fitness_scores: list):
        current_time = time.time()
        # Output status dashboard every 60 seconds for optimal feedback loops
        if current_time - self.last_print_time >= 60.0:  
            elapsed_time = current_time - self.start_time
            fps = int(current_steps / elapsed_time) if elapsed_time > 0 else 0
            frame_fps = int((current_steps * self.frame_skip) / elapsed_time) if elapsed_time > 0 else 0
            decision_fps = int((current_steps * self.num_agents) / elapsed_time) if elapsed_time > 0 else 0
            progress_pct = (current_steps / self.total_steps) * 100
            remaining_steps = self.total_steps - current_steps
            eta_hours = (remaining_steps / fps) / 3600 if fps > 0 else 0
            
            best_fit = max(fitness_scores) if fitness_scores else float('-inf')
            avg_fit = np.mean(fitness_scores) if fitness_scores else float('-inf')

            print(
                f"🧬 [EVO TELEMETRY] Steps: {current_steps:,}/{self.total_steps:,} ({progress_pct:.2f}%) | "
                f"Env Steps/s: {fps} | Atari Frames/s~: {frame_fps} | Decisions/s~: {decision_fps} | "
                f"Best Generation Fit: {best_fit:.2f} | Avg Fit: {avg_fit:.2f} | "
                f"ETA: {eta_hours:.2f}h"
            )
            self.last_print_time = current_time


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def _safe_std(values):
    return float(np.std(values)) if values else 0.0


def _normalized_action_entropy(action_counts):
    total_actions = int(np.sum(action_counts))
    if total_actions == 0:
        return 0.0

    probs = action_counts / total_actions
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0

    entropy = float(-np.sum(probs * np.log(probs)))
    max_entropy = float(np.log(len(action_counts))) if len(action_counts) > 1 else 1.0
    if max_entropy <= 0:
        return 0.0
    return entropy / max_entropy


def _latest_frame(obs):
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
    frame = _latest_frame(obs)
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


def evaluate_brain_vs_heuristic(brain, device, episodes=4):
    eval_env = make_agile_pong_env()
    eval_agents = eval_env.possible_agents
    model_agent = "second_0"
    opponent_agent = "first_0"
    if model_agent not in eval_agents or opponent_agent not in eval_agents:
        model_agent = eval_agents[0]
        opponent_agent = eval_agents[1]

    rewards = []
    try:
        for _ in range(episodes):
            states, _ = eval_env.reset()
            episode_reward = 0.0

            while eval_env.agents:
                actions = {}

                if model_agent in eval_env.agents:
                    state_tensor = torch.from_numpy(np.array(states[model_agent])).unsqueeze(0).to(device)
                    action, _, _, _ = brain.get_action(state_tensor)
                    actions[model_agent] = int(np.array(action).squeeze())

                if opponent_agent in eval_env.agents:
                    actions[opponent_agent] = heuristic_pong_action(states[opponent_agent], side="right")

                states, step_rewards, _, _, _ = eval_env.step(actions)
                if model_agent in step_rewards:
                    episode_reward += float(step_rewards[model_agent])

            rewards.append(episode_reward)
    finally:
        eval_env.close()

    return _safe_mean(rewards), _safe_std(rewards)


def evaluate_brain_vs_sb3(brain, sb3_model, device, episodes=4):
    eval_env = make_agile_pong_env()
    eval_agents = eval_env.possible_agents
    model_agent = "second_0"
    opponent_agent = "first_0"
    if model_agent not in eval_agents or opponent_agent not in eval_agents:
        model_agent = eval_agents[0]
        opponent_agent = eval_agents[1]

    rewards = []
    try:
        for _ in range(episodes):
            states, _ = eval_env.reset()
            episode_reward = 0.0

            while eval_env.agents:
                actions = {}

                if model_agent in eval_env.agents:
                    state_tensor = torch.as_tensor(states[model_agent], device=device).unsqueeze(0)
                    action, _, _, _ = brain.get_action(state_tensor)
                    actions[model_agent] = int(np.array(action).squeeze())

                if opponent_agent in eval_env.agents:
                    sb3_input = np.expand_dims(np.asarray(states[opponent_agent]), axis=0)
                    sb3_action, _ = sb3_model.predict(sb3_input, deterministic=True)
                    actions[opponent_agent] = int(sb3_action[0])

                states, step_rewards, _, _, _ = eval_env.step(actions)
                if model_agent in step_rewards:
                    episode_reward += float(step_rewards[model_agent])

            rewards.append(episode_reward)
    finally:
        eval_env.close()

    return _safe_mean(rewards), _safe_std(rewards)


def _new_pending_rollout(brain):
    return {
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


def _merge_rollout_into_pending(pending, episode_rollout):
    pending["observations"].extend(episode_rollout["observations"])
    pending["actions"].extend(episode_rollout["actions"])
    pending["log_probs"].extend(episode_rollout["log_probs"])
    pending["rewards"].extend(episode_rollout["rewards"])
    pending["dones"].extend(episode_rollout["dones"])
    pending["values"].extend(episode_rollout["values"])
    pending["next_obs"] = episode_rollout["next_obs"]
    pending["next_done"] = episode_rollout["next_done"]


def _flush_pending_rollout(pending):
    if not pending["rewards"]:
        return False

    experiences = (
        pending["observations"],
        pending["actions"],
        pending["log_probs"],
        pending["rewards"],
        pending["dones"],
        pending["values"],
        pending["next_obs"],
        pending["next_done"],
    )
    pending["brain"].learn(experiences)
    pending.update(_new_pending_rollout(pending["brain"]))
    return True


def _current_sb3_mix_prob(start_prob, end_prob, ramp_generations, current_generation):
    if ramp_generations <= 0:
        return end_prob
    progress = min(max(current_generation, 0) / float(ramp_generations), 1.0)
    return float(start_prob + (end_prob - start_prob) * progress)


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

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    VERSION_TAG = "agile_marl_v1"
    MASTER_RUNS_DIR = "./training_runs"
    SINGLE_AGENT_PATH = "./training_runs/single_agent/ppo_pong_model.zip"
    version_dir = os.path.join(MASTER_RUNS_DIR, VERSION_TAG)
    checkpoint_dir = os.path.join(version_dir, "evolved_brains")
    metrics_path = os.path.join(version_dir, "selfplay_metrics.txt")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"📁 Saving evolutionary checkpoints cleanly to: {checkpoint_dir}\n")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("SELFPLAY RUN METRICS\n")
        f.write(f"start_time_unix={int(time.time())}\n")
        f.write(f"version_tag={VERSION_TAG}\n")
        f.write(f"device={device}\n")

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
        "no_mut": 0.7,
        "architect_mut": 0.05,
        "param_mut": 0.2,                  
        "lr_mut": 0.05,
        "mutation_strength": 0.02,
    }

    POP_SIZE = 8
    EXTERNAL_EVAL_EPISODES = 4
    EXTERNAL_EVAL_INTERVAL_GENS = 5
    SB3_EVAL_EPISODES = 8
    SB3_EVAL_INTERVAL_GENS = 3
    SB3_MIX_PROB_START = 0.15
    SB3_MIX_PROB_END = 0.45
    SB3_MIX_RAMP_GENS = 30
    FRAME_SKIP = 4
    LEARN_MIN_TRANSITIONS = 4096
    print(f"Spawning genetic population pool consisting of {POP_SIZE} active strains...")

    sb3_model = None
    sb3_enabled = os.path.exists(SINGLE_AGENT_PATH)
    if sb3_enabled:
        sb3_model = load_sb3_model_compat(
            SINGLE_AGENT_PATH,
            observation_space=init_env.observation_space(agents[0]),
            action_space=init_env.action_space(agents[0]),
        )
        print(f"🤖 SB3 sparring partner enabled from: {SINGLE_AGENT_PATH}")
    else:
        print(f"⚠️ SB3 sparring disabled. Missing model at: {SINGLE_AGENT_PATH}")
    
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
        tournament_size=4,
        elitism=True,                      
        population_size=POP_SIZE,          
        eval_loop=3,
    )

    mutation_manager = Mutations(
        no_mutation=MUTATION_PARAMS["no_mut"],
        architecture=MUTATION_PARAMS["architect_mut"],
        new_layer_prob=0.2,
        parameters=MUTATION_PARAMS["param_mut"],
        activation=0.0,
        rl_hp=MUTATION_PARAMS["lr_mut"],
        mutation_sd=MUTATION_PARAMS["mutation_strength"],
        mutate_elite=False,
        device=device,
    )

    env = make_agile_pong_env()
    
    TOTAL_TARGET_STEPS = 10_000_000
    MAX_HOURS = 12.0
    EVO_STEP_INTERVAL = 100_000        # Every 100,000 steps triggers tournament updates
    
    global_step_counter = 0
    next_evo_milestone = EVO_STEP_INTERVAL
    evo_generation_count = 0

    generation_match_rewards_left = []
    generation_match_rewards_right = []
    generation_match_lengths = []
    generation_sb3_mix_matches = 0
    generation_selfplay_matches = 0
    generation_sb3_model_rewards = []
    generation_action_counts = np.zeros(action_dim, dtype=np.int64)
    
    # Track historical fitness scores for each strain to enable proper selection
    brain_fitness_trackers = {id(brain): [] for brain in population}
    pending_rollouts = {id(brain): _new_pending_rollout(brain) for brain in population}
    
    progress_monitor = AgileProgressMonitor(
        total_steps=TOTAL_TARGET_STEPS,
        frame_skip=FRAME_SKIP,
        num_agents=len(agents),
    )
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
            sb3_mix_prob = _current_sb3_mix_prob(
                SB3_MIX_PROB_START,
                SB3_MIX_PROB_END,
                SB3_MIX_RAMP_GENS,
                evo_generation_count,
            )
            use_sb3_opponent = bool(sb3_enabled and np.random.random() < sb3_mix_prob)
            learner_agent_id = None
            sb3_agent_id = None
            if use_sb3_opponent:
                generation_sb3_mix_matches += 1
            else:
                generation_selfplay_matches += 1

            episode_brains = {}
            if use_sb3_opponent:
                learner_agent_id = np.random.choice(agents)
                sb3_agent_candidates = [agent for agent in agents if agent != learner_agent_id]
                sb3_agent_id = sb3_agent_candidates[0]
                for agent_id in agents:
                    if agent_id == learner_agent_id:
                        episode_brains[agent_id] = np.random.choice(population)
                    else:
                        episode_brains[agent_id] = None
            else:
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
                if brain is not None
            }
            episode_length = 0
            
            while env.agents:
                actions = {}
                for agent_id in env.agents:
                    active_brain = episode_brains[agent_id]
                    if active_brain is None:
                        sb3_input = np.expand_dims(np.asarray(states[agent_id]), axis=0)
                        sb3_action, _ = sb3_model.predict(sb3_input, deterministic=True)
                        selected_action = int(sb3_action[0])
                    else:
                        state_tensor = torch.as_tensor(states[agent_id], device=device).unsqueeze(0)
                        action, log_prob, _, value = active_brain.get_action(state_tensor)

                        brain_rollout = episode_rollouts[id(active_brain)]
                        brain_rollout["observations"].append(np.asarray(states[agent_id]))
                        brain_rollout["actions"].append(np.array(action).squeeze())
                        brain_rollout["log_probs"].append(np.array(log_prob).squeeze())
                        brain_rollout["values"].append(np.array(value).squeeze())

                        selected_action = int(np.array(action).squeeze())

                    actions[agent_id] = selected_action
                    if 0 <= selected_action < action_dim:
                        generation_action_counts[selected_action] += 1

                next_states, rewards, terminations, truncations, infos = env.step(actions)
                
                for agent_id in rewards.keys():
                    active_brain = episode_brains[agent_id]
                    done = terminations[agent_id] or truncations[agent_id]
                    episode_rewards[agent_id] += rewards[agent_id]

                    if active_brain is None:
                        continue

                    brain_rollout = episode_rollouts[id(active_brain)]
                    brain_rollout["rewards"].append(np.array(rewards[agent_id], dtype=np.float32))
                    brain_rollout["dones"].append(np.array(done, dtype=np.float32))
                    brain_rollout["next_obs"] = np.asarray(next_states[agent_id])
                    brain_rollout["next_done"] = np.array(done, dtype=np.float32)

                # Count one environment transition per parallel step.
                global_step_counter += 1
                episode_length += 1

                states = next_states
            
            # Record match results to update running fitness averages
            for agent_id, reward in episode_rewards.items():
                active_brain = episode_brains[agent_id]
                if active_brain is None:
                    continue
                brain_fitness_trackers[id(active_brain)].append(reward)

            if use_sb3_opponent and learner_agent_id is not None and sb3_agent_id is not None:
                generation_sb3_model_rewards.append(float(episode_rewards[learner_agent_id]))
            
            if use_sb3_opponent and learner_agent_id is not None and sb3_agent_id is not None:
                print(
                    f"🎮 [MATCH END] Global Steps: {global_step_counter:,} | "
                    f"Mode: SB3_MIX(p={sb3_mix_prob:.2f}) | "
                    f"Learner[{learner_agent_id}]: {episode_rewards[learner_agent_id]:.1f} | "
                    f"SB3[{sb3_agent_id}]: {episode_rewards[sb3_agent_id]:.1f}"
                )
            else:
                print(
                    f"🎮 [MATCH END] Global Steps: {global_step_counter:,} | "
                    f"Mode: SELF_PLAY | "
                    f"{agents[0]}: {episode_rewards[agents[0]]:.1f} | "
                    f"{agents[1]}: {episode_rewards[agents[1]]:.1f}"
                )

            generation_match_rewards_left.append(float(episode_rewards[agents[0]]))
            generation_match_rewards_right.append(float(episode_rewards[agents[1]]))
            generation_match_lengths.append(int(episode_length))

            # --- STEP 2: BRAIN OPTIMIZATION GRADIENTS ---
            for rollout in episode_rollouts.values():
                brain_id = id(rollout["brain"])
                pending = pending_rollouts[brain_id]
                _merge_rollout_into_pending(pending, rollout)

            for pending in pending_rollouts.values():
                if len(pending["rewards"]) >= LEARN_MIN_TRANSITIONS:
                    _flush_pending_rollout(pending)

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
                
                # Display structural details about your neural networks' hyperparameters
                print("📋 Current Population Metrics:")
                for idx, brain in enumerate(population):
                    current_lr = brain.lr if hasattr(brain, 'lr') else INIT_HP["LR"]
                    print(f"  🔹 Strain {idx+1} | Mean Match Reward: {brain.fitness[-1]:.2f} | Learning Rate: {current_lr}")
                
                # Execute tournament selection then apply mutation operators.
                elite_agent, population = evolutionary_manager.select(population)
                population = mutation_manager.mutation(population)

                # Reset fitness trackers for the newly cloned/mutated generation.
                brain_fitness_trackers = {id(brain): [] for brain in population}
                pending_rollouts = {id(brain): _new_pending_rollout(brain) for brain in population}

                # Save the true elite from pre-mutation evaluation.
                save_path = os.path.join(checkpoint_dir, f"pong_champ_gen_{evo_generation_count}.pt")
                elite_agent.save_checkpoint(save_path)

                heuristic_eval_mean = float("nan")
                heuristic_eval_std = float("nan")
                sb3_eval_mean = float("nan")
                sb3_eval_std = float("nan")
                if evo_generation_count % EXTERNAL_EVAL_INTERVAL_GENS == 0:
                    # External validation is expensive, so evaluate periodically.
                    heuristic_eval_mean, heuristic_eval_std = evaluate_brain_vs_heuristic(
                        elite_agent,
                        device,
                        episodes=EXTERNAL_EVAL_EPISODES,
                    )
                    print(
                        f"🧪 [EXTERNAL EVAL] Elite vs heuristic | "
                        f"mean_reward={heuristic_eval_mean:.3f} std={heuristic_eval_std:.3f}"
                    )

                if sb3_enabled and evo_generation_count % SB3_EVAL_INTERVAL_GENS == 0:
                    sb3_eval_mean, sb3_eval_std = evaluate_brain_vs_sb3(
                        elite_agent,
                        sb3_model,
                        device,
                        episodes=SB3_EVAL_EPISODES,
                    )
                    print(
                        f"🧪 [EXTERNAL EVAL] Elite vs SB3 | "
                        f"mean_reward={sb3_eval_mean:.3f} std={sb3_eval_std:.3f}"
                    )

                # Assemble educational generation metrics to understand training dynamics.
                wins_left = int(
                    np.sum(np.array(generation_match_rewards_left) > np.array(generation_match_rewards_right))
                ) if generation_match_rewards_left else 0
                wins_right = int(
                    np.sum(np.array(generation_match_rewards_right) > np.array(generation_match_rewards_left))
                ) if generation_match_rewards_right else 0
                draws = int(len(generation_match_rewards_left) - wins_left - wins_right)

                action_total = int(np.sum(generation_action_counts))
                action_distribution = (
                    (generation_action_counts / action_total).round(4).tolist() if action_total > 0
                    else [0.0] * action_dim
                )

                mutation_labels = {}
                for brain in population:
                    label = getattr(brain, "mut", "unknown")
                    mutation_labels[label] = mutation_labels.get(label, 0) + 1

                generation_metrics = {
                    "generation": int(evo_generation_count),
                    "global_steps": int(global_step_counter),
                    "wall_clock_seconds": float(time.time() - start_wall_clock),
                    "matches_this_generation": int(len(generation_match_lengths)),
                    "sb3_mix_matches_this_generation": int(generation_sb3_mix_matches),
                    "selfplay_matches_this_generation": int(generation_selfplay_matches),
                    "sb3_mix_prob_current": float(sb3_mix_prob),
                    "mean_match_length": _safe_mean(generation_match_lengths),
                    "std_match_length": _safe_std(generation_match_lengths),
                    "sb3_mix_model_reward_mean": _safe_mean(generation_sb3_model_rewards),
                    "sb3_mix_model_reward_std": _safe_std(generation_sb3_model_rewards),
                    "left_reward_mean": _safe_mean(generation_match_rewards_left),
                    "left_reward_std": _safe_std(generation_match_rewards_left),
                    "right_reward_mean": _safe_mean(generation_match_rewards_right),
                    "right_reward_std": _safe_std(generation_match_rewards_right),
                    "win_rate_left": (wins_left / len(generation_match_lengths)) if generation_match_lengths else 0.0,
                    "win_rate_right": (wins_right / len(generation_match_lengths)) if generation_match_lengths else 0.0,
                    "draw_rate": (draws / len(generation_match_lengths)) if generation_match_lengths else 0.0,
                    "population_fitness_mean": _safe_mean([brain.fitness[-1] for brain in population if brain.fitness]),
                    "population_fitness_std": _safe_std([brain.fitness[-1] for brain in population if brain.fitness]),
                    "elite_fitness": float(elite_agent.fitness[-1]) if elite_agent.fitness else 0.0,
                    "learning_rates": [float(getattr(brain, "lr", INIT_HP["LR"])) for brain in population],
                    "mutation_labels": mutation_labels,
                    "action_distribution": action_distribution,
                    "action_entropy_norm": _normalized_action_entropy(generation_action_counts),
                    "elite_vs_heuristic_mean": float(heuristic_eval_mean),
                    "elite_vs_heuristic_std": float(heuristic_eval_std),
                    "elite_vs_sb3_mean": float(sb3_eval_mean),
                    "elite_vs_sb3_std": float(sb3_eval_std),
                }

                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write("\n--- GENERATION SUMMARY ---\n")
                    f.write(
                        f"gen={generation_metrics['generation']} steps={generation_metrics['global_steps']} "
                        f"matches={generation_metrics['matches_this_generation']} "
                        f"sb3_mix={generation_metrics['sb3_mix_matches_this_generation']} "
                        f"selfplay={generation_metrics['selfplay_matches_this_generation']} "
                        f"sb3_mix_prob={generation_metrics['sb3_mix_prob_current']:.2f} "
                        f"sb3_mix_model_mean={generation_metrics['sb3_mix_model_reward_mean']:.3f} "
                        f"left_mean={generation_metrics['left_reward_mean']:.3f} "
                        f"right_mean={generation_metrics['right_reward_mean']:.3f} "
                        f"elite_fit={generation_metrics['elite_fitness']:.3f} "
                        f"entropy_norm={generation_metrics['action_entropy_norm']:.3f} "
                        f"heuristic_mean={generation_metrics['elite_vs_heuristic_mean']:.3f} "
                        f"sb3_mean={generation_metrics['elite_vs_sb3_mean']:.3f}\n"
                    )
                    f.write(json.dumps(generation_metrics) + "\n")

                # Reset per-generation telemetry buffers after logging.
                generation_match_rewards_left = []
                generation_match_rewards_right = []
                generation_match_lengths = []
                generation_sb3_mix_matches = 0
                generation_selfplay_matches = 0
                generation_sb3_model_rewards = []
                generation_action_counts = np.zeros(action_dim, dtype=np.int64)

                print(f"💾 Saved elite genetic snapshot cleanly to: {save_path}\n")
                
                # Target the next sequential milestone interval bound safely
                next_evo_milestone += EVO_STEP_INTERVAL

            # Run regular 60-second console dashboard telemetry updates
            current_generation_fitness = [np.mean(brain_fitness_trackers[id(b)]) if brain_fitness_trackers[id(b)] else 0.0 for b in population]
            progress_monitor.check_and_print(global_step_counter, current_generation_fitness)

    except KeyboardInterrupt:
        print("\nExecution broken manually by keyboard. Wrapping up files safely...")
    finally:
        for pending in pending_rollouts.values():
            _flush_pending_rollout(pending)
        env.close()
        print(f"\n========================================================")
        print(f"--- AGILE_MARL RUN COMPLETE ---")
        print(f"Total Combined Step Counts Achieved: {global_step_counter:,}")
        print(f"Target Checkpoint Location: {checkpoint_dir}")
        print(f"========================================================\n")

if __name__ == "__main__":
    main()