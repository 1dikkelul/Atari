import os
import copy
import torch
import torch.nn as nn
import numpy as np
from agilerl.utils.utils import create_population
from gymnasium import spaces

# --- CONFIGURATIONS ---
MODEL_PATH = "./training_runs/agile_marl_v1/evolved_brains/pong_champ_gen_30.pt"
OUTPUT_ONNX_PATH = "./public/models/pong_champ_gen_30.onnx"

def export_agilerl_to_onnx():
    obs_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
    act_space = spaces.Discrete(6)
    
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

    print("Initializing dummy AgileRL architecture...")
    dummy_pop = create_population(
        algo=INIT_HP["ALGO"],
        observation_space=obs_space,
        action_space=act_space,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        population_size=1,
        device="cpu"
    )
    agent = dummy_pop[0]
    
    print(f"Loading trained weights from {MODEL_PATH}...")
    agent.load_checkpoint(MODEL_PATH)
    
    print("Constructing matching standard PyTorch Sequential architecture...")
    # Build export pipeline from the checkpoint-loaded actor modules so evolved
    # layer widths/depth are mirrored exactly.
    encoder = agent.actor.encoder.model
    head_model = agent.actor.head_net._wrapped.model

    modules = [
        copy.deepcopy(encoder.actor_encoder_conv_layer_1),
        nn.ReLU(),
        copy.deepcopy(encoder.actor_encoder_conv_layer_2),
        nn.ReLU(),
        copy.deepcopy(encoder.actor_encoder_conv_layer_3),
        nn.ReLU(),
        nn.Flatten(start_dim=1),
        copy.deepcopy(encoder.actor_encoder_linear_output),
        nn.ReLU(),
    ]

    hidden_idx = 1
    while hasattr(head_model, f"actor_linear_layer_{hidden_idx}"):
        modules.append(copy.deepcopy(getattr(head_model, f"actor_linear_layer_{hidden_idx}")))

        layer_norm_name = f"actor_layer_norm_{hidden_idx}"
        if hasattr(head_model, layer_norm_name):
            modules.append(copy.deepcopy(getattr(head_model, layer_norm_name)))

        modules.append(nn.ReLU())
        hidden_idx += 1

    modules.append(copy.deepcopy(head_model.actor_linear_layer_output))
    clean_pipeline = nn.Sequential(*modules)
    clean_pipeline.eval()

    # Create dummy tensor matching (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 4, 84, 84, dtype=torch.float32)

    os.makedirs(os.path.dirname(OUTPUT_ONNX_PATH), exist_ok=True)

    print(f"Exporting clean sequential tensor path to ONNX at: {OUTPUT_ONNX_PATH}...")
    
    torch.onnx.export(
        clean_pipeline,
        dummy_input,
        OUTPUT_ONNX_PATH,
        export_params=True,
        opset_version=18,          
        do_constant_folding=True,  
        input_names=['input'],     
        output_names=['output'],   
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("✅ Export complete! Model weights successfully decoupled into standard ONNX.")

if __name__ == "__main__":
    export_agilerl_to_onnx()