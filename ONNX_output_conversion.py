import os
import torch
import torch.nn as nn
import numpy as np
from agilerl.utils.utils import create_population
from gymnasium import spaces

# --- CONFIGURATIONS ---
MODEL_PATH = "./training_runs/evolutionary_league/pong_champ_gen_100.pt"
OUTPUT_ONNX_PATH = "./public/models/pong_champ_gen_100.onnx"

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
    
    # Grab the raw state dict containing all the actual numeric weights
    src_state = agent.actor.state_dict()

    print("Constructing matching standard PyTorch Sequential architecture...")
    # Rebuilding the exact network topology using native PyTorch elements
    clean_pipeline = nn.Sequential(
        # --- CNN Encoder Section ---
        nn.Conv2d(4, 32, kernel_size=8, stride=4), # actor_encoder_conv_layer_1
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2), # actor_encoder_conv_layer_2
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1), # actor_encoder_conv_layer_3
        nn.ReLU(),
        nn.Flatten(start_dim=1),
        
        # --- Internal Encoder Compression Layer ---
        nn.Linear(3136, 32), # actor_encoder_linear_output
        nn.ReLU(),
        
        # --- Head Net Section ---
        nn.Linear(32, 32), # actor_linear_layer_1
        nn.LayerNorm(32),   # actor_layer_norm_1
        nn.ReLU(),
        nn.Linear(32, 6)   # actor_linear_layer_output (6 actions)
    )

    # Map the structural tensor weights explicitly from AgileRL keys to the standard layers
    target_state = clean_pipeline.state_dict()
    
    mapping = {
        '0.weight': 'encoder.model.actor_encoder_conv_layer_1.weight',
        '0.bias':   'encoder.model.actor_encoder_conv_layer_1.bias',
        '2.weight': 'encoder.model.actor_encoder_conv_layer_2.weight',
        '2.bias':   'encoder.model.actor_encoder_conv_layer_2.bias',
        '4.weight': 'encoder.model.actor_encoder_conv_layer_3.weight',
        '4.bias':   'encoder.model.actor_encoder_conv_layer_3.bias',
        '7.weight': 'encoder.model.actor_encoder_linear_output.weight',
        '7.bias':   'encoder.model.actor_encoder_linear_output.bias',
        '9.weight': 'head_net._wrapped.model.actor_linear_layer_1.weight',
        '9.bias':   'head_net._wrapped.model.actor_linear_layer_1.bias',
        '10.weight':'head_net._wrapped.model.actor_layer_norm_1.weight',
        '10.bias':  'head_net._wrapped.model.actor_layer_norm_1.bias',
        '12.weight':'head_net._wrapped.model.actor_linear_layer_output.weight',
        '12.bias':  'head_net._wrapped.model.actor_linear_layer_output.bias',
    }

    # Inject the weights into our clean pipeline
    new_state_dict = {}
    for target_key, src_key in mapping.items():
        new_state_dict[target_key] = src_state[src_key]
        
    clean_pipeline.load_state_dict(new_state_dict)
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