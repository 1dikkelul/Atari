import argparse
import copy
import os
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import torch
from gymnasium import spaces
from pettingzoo.atari import pong_v3
import supersuit as ss

from agilerl.utils.utils import create_population


DEFAULT_CHECKPOINT = "./training_runs/agile_marl_v1/evolved_brains/pong_champ_gen_70.pt"
DEFAULT_ONNX = "./public/models/pong_champ_gen_70.onnx"


class ChannelFirstParallelWrapper:
    def __init__(self, env):
        self.env = env
        self.possible_agents = env.possible_agents

    @property
    def agents(self):
        return self.env.agents

    def _transpose_obs(self, observations):
        return {agent: np.transpose(obs, (2, 0, 1)) for agent, obs in observations.items()}

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
    env = pong_v3.parallel_env(obs_type="rgb_image", max_cycles=100000)
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.color_reduction_v0(env, mode="full")
    env = ss.resize_v1(env, 84, 84)
    env = ss.frame_stack_v2(env, 4)
    env = ChannelFirstParallelWrapper(env)
    return env


def load_checkpoint_agent(checkpoint_path, device):
    obs_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
    act_space = spaces.Discrete(6)

    net_config = {
        "encoder_config": {
            "channel_size": [32, 64, 64],
            "kernel_size": [8, 4, 3],
            "stride_size": [4, 2, 1],
        }
    }
    init_hp = {
        "ALGO": "PPO",
        "BATCH_SIZE": 256,
        "LR": 2.5e-4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "SHARE_ENCODERS": False,
        "ACTION_MASKED": False,
    }

    population = create_population(
        algo=init_hp["ALGO"],
        observation_space=obs_space,
        action_space=act_space,
        net_config=net_config,
        INIT_HP=init_hp,
        population_size=1,
        device=device,
    )
    agent = population[0]
    agent.load_checkpoint(checkpoint_path)
    return agent


def build_export_pipeline(agent):
    # Mirror ONNX_output_conversion.py so we can check parity at the same boundary.
    encoder = agent.actor.encoder.model
    head_model = agent.actor.head_net._wrapped.model

    modules = [
        copy.deepcopy(encoder.actor_encoder_conv_layer_1),
        torch.nn.ReLU(),
        copy.deepcopy(encoder.actor_encoder_conv_layer_2),
        torch.nn.ReLU(),
        copy.deepcopy(encoder.actor_encoder_conv_layer_3),
        torch.nn.ReLU(),
        torch.nn.Flatten(start_dim=1),
        copy.deepcopy(encoder.actor_encoder_linear_output),
        torch.nn.ReLU(),
    ]

    hidden_idx = 1
    while hasattr(head_model, f"actor_linear_layer_{hidden_idx}"):
        modules.append(copy.deepcopy(getattr(head_model, f"actor_linear_layer_{hidden_idx}")))

        layer_norm_name = f"actor_layer_norm_{hidden_idx}"
        if hasattr(head_model, layer_norm_name):
            modules.append(copy.deepcopy(getattr(head_model, layer_norm_name)))

        modules.append(torch.nn.ReLU())
        hidden_idx += 1

    modules.append(copy.deepcopy(head_model.actor_linear_layer_output))
    model = torch.nn.Sequential(*modules)
    model.eval()
    return model


def sample_observations(env, num_samples, seed):
    rng = np.random.default_rng(seed)
    observations = []

    states, _ = env.reset(seed=seed)
    agents = env.possible_agents

    while len(observations) < num_samples:
        for agent_id in agents:
            if agent_id in states:
                observations.append(np.array(states[agent_id], copy=True))
                if len(observations) >= num_samples:
                    break

        if len(observations) >= num_samples:
            break

        actions = {}
        for agent_id in env.agents:
            action_n = env.action_space(agent_id).n
            # Mix random and NOOP-heavy actions to get varied yet realistic states.
            if rng.random() < 0.35:
                action = 0
            else:
                action = int(rng.integers(0, action_n))
            actions[agent_id] = action

        next_states, _, terminations, truncations, _ = env.step(actions)

        if all(terminations.values()) or all(truncations.values()) or len(env.agents) == 0:
            states, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        else:
            states = next_states

    return observations


@dataclass
class ParitySummary:
    num_samples: int
    onnx_argmax_counts: list
    torch_argmax_counts: list
    agent_sample_counts: list
    argmax_match_rate: float
    max_abs_diff: float
    mean_abs_diff: float


def run_parity_check(checkpoint_path, onnx_path, num_samples, seed, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    agent = load_checkpoint_agent(checkpoint_path=checkpoint_path, device=device)
    torch_model = build_export_pipeline(agent)

    # Keep AgileRL internals and tensors on one device for get_action sampling.
    device_obj = torch.device(device)
    if hasattr(agent, "actor"):
        agent.actor.to(device_obj)
    if hasattr(agent, "critic"):
        agent.critic.to(device_obj)
    if hasattr(agent, "device"):
        agent.device = device

    # CPU provider is deterministic and avoids GPU provider setup issues for diagnostics.
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    env = make_agile_pong_env()
    try:
        sampled_obs = sample_observations(env, num_samples=num_samples, seed=seed)
    finally:
        env.close()

    torch_argmax = []
    onnx_argmax = []
    agent_samples = []
    abs_diffs = []

    for obs in sampled_obs:
        x = obs.astype(np.float32, copy=False)
        x_batch = np.expand_dims(x, axis=0)

        with torch.no_grad():
            torch_out = torch_model(torch.from_numpy(x_batch)).cpu().numpy()

        onnx_out = session.run([output_name], {input_name: x_batch})[0]

        # This is the actual action API used by play_pong.py and selfplay.py.
        state_tensor = torch.from_numpy(x_batch).to(device_obj)
        sampled_action, _, _, _ = agent.get_action(state_tensor)
        sampled_idx = int(np.array(sampled_action).squeeze())

        torch_idx = int(np.argmax(torch_out[0]))
        onnx_idx = int(np.argmax(onnx_out[0]))

        torch_argmax.append(torch_idx)
        onnx_argmax.append(onnx_idx)
        agent_samples.append(sampled_idx)

        abs_diff = np.abs(torch_out - onnx_out)
        abs_diffs.append(abs_diff)

    diff_stack = np.concatenate([d.reshape(1, -1) for d in abs_diffs], axis=0)

    torch_counts = np.bincount(np.array(torch_argmax), minlength=6).tolist()
    onnx_counts = np.bincount(np.array(onnx_argmax), minlength=6).tolist()
    agent_sample_counts = np.bincount(np.array(agent_samples), minlength=6).tolist()
    match_rate = float(np.mean(np.array(torch_argmax) == np.array(onnx_argmax)))

    return ParitySummary(
        num_samples=num_samples,
        onnx_argmax_counts=onnx_counts,
        torch_argmax_counts=torch_counts,
        agent_sample_counts=agent_sample_counts,
        argmax_match_rate=match_rate,
        max_abs_diff=float(np.max(diff_stack)),
        mean_abs_diff=float(np.mean(diff_stack)),
    )


def main():
    parser = argparse.ArgumentParser(description="Parity check between checkpoint-derived PyTorch head and ONNX export")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to .pt checkpoint")
    parser.add_argument("--onnx", default=DEFAULT_ONNX, help="Path to .onnx model")
    parser.add_argument("--samples", type=int, default=256, help="Number of state samples to compare")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for state sampling")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for loading checkpoint model",
    )

    args = parser.parse_args()

    summary = run_parity_check(
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx,
        num_samples=args.samples,
        seed=args.seed,
        device=args.device,
    )

    print("=" * 72)
    print("ONNX PARITY CHECK SUMMARY")
    print("=" * 72)
    print(f"checkpoint: {args.checkpoint}")
    print(f"onnx:       {args.onnx}")
    print(f"samples:    {summary.num_samples}")
    print("-" * 72)
    print(f"argmax match rate: {summary.argmax_match_rate:.4f}")
    print(f"max abs diff:      {summary.max_abs_diff:.6e}")
    print(f"mean abs diff:     {summary.mean_abs_diff:.6e}")
    print("-" * 72)
    print(f"torch argmax counts: {summary.torch_argmax_counts}")
    print(f"onnx argmax counts:  {summary.onnx_argmax_counts}")
    print(f"agent sampled counts: {summary.agent_sample_counts}")

    onnx_total = max(1, sum(summary.onnx_argmax_counts))
    onnx_dominant_idx = int(np.argmax(np.array(summary.onnx_argmax_counts)))
    onnx_dominant_ratio = summary.onnx_argmax_counts[onnx_dominant_idx] / onnx_total
    print(f"onnx dominant action index: {onnx_dominant_idx} ({onnx_dominant_ratio:.2%})")

    print("=" * 72)
    if summary.argmax_match_rate < 0.95:
        print("Result: MISMATCH likely introduced during conversion/export.")
    else:
        print("Result: Conversion looks consistent.")
        if len(set(np.nonzero(summary.agent_sample_counts)[0].tolist())) > 1:
            print("Note: Agent sampling uses multiple actions while argmax may still be single-action.")
            print("      Browser ONNX currently uses deterministic argmax, so behavior can look collapsed.")
        else:
            print("Note: Agent sampling is also concentrated. Next suspect is training/policy collapse.")
    print("=" * 72)


if __name__ == "__main__":
    main()
