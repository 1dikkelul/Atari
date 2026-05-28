import os
import torch
import ale_py
from stable_baselines3.common.env_util import make_atari_env

def inspect_env_stack(env_id):
    print(f"\n==================================================")
    print(f"INSPECTING WRAPPER STACK FOR: {env_id}")
    print(f"==================================================")
    
    # Create just 1 environment instance using the SB3 factory method
    env = make_atari_env(env_id, n_envs=1, seed=42)
    
    # Pull out the very first environment vector
    current_layer = env.envs[0]
    
    # Walk down the nesting doll chain
    layer_idx = 1
    sb3_skip = None
    
    while hasattr(current_layer, 'env'):
        wrapper_name = type(current_layer).__name__
        print(f"Layer {layer_idx}: {wrapper_name}")
        
        # Capture the explicit skip setting inside SB3's custom wrapper
        if wrapper_name == "MaxAndSkipEnv":
            sb3_skip = getattr(current_layer, '_skip', 'Unknown')
            print(f"  ---> Found SB3 Frame Skip! Value: {sb3_skip}")
            
        current_layer = current_layer.env
        layer_idx += 1
        
    # We reached the bottom! Print the raw underlying emulator environment
    base_env_name = type(current_layer).__name__
    print(f"Base Core Layer: {base_env_name}")
    
    # Look at the raw Gymnasium spec dictionary to see what the emulator baked in
    base_skip = "N/A"
    if hasattr(current_layer, 'spec') and current_layer.spec is not None:
        base_skip = current_layer.spec.kwargs.get('frameskip', 4) 
        print(f"  ---> Found Base Emulator Frameskip! Value: {base_skip}")
        
    print("-" * 50)
    
    # Calculate total stacked skipping math
    if sb3_skip is not None and isinstance(sb3_skip, int) and isinstance(base_skip, int):
        total_skip = sb3_skip * base_skip
        print(f"👉 MATHEMATICAL RESULT: 1 step in your code moves the game forward by {total_skip} raw frames.")
    else:
        print(f"👉 MATHEMATICAL RESULT: Standard 1 layer skip of {base_skip if sb3_skip is None else sb3_skip} raw frames.")
    print(f"==================================================\n")
    
    env.close()

if __name__ == "__main__":
    # Test both to see the hidden contrast live on your terminal!
    inspect_env_stack("ALE/Pong-v5")
    inspect_env_stack("PongNoFrameskip-v4")