import numpy as np
from agilerl.utils.utils import create_population
from gymnasium import spaces

obs_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
act_space = spaces.Discrete(6)

NET_CONFIG = {
    "encoder_config": {
        "channel_size": [32, 64, 64], "kernel_size": [8, 4, 3], "stride_size": [4, 2, 1]
    }
}
INIT_HP = {
    "ALGO": "PPO", "BATCH_SIZE": 256, "LR": 2.5e-4, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
    "SHARE_ENCODERS": False, "ACTION_MASKED": False
}

dummy_pop = create_population(
    algo=INIT_HP["ALGO"], observation_space=obs_space, action_space=act_space,
    net_config=NET_CONFIG, INIT_HP=INIT_HP, population_size=1, device="cpu"
)

# Print out exactly what parameters and modules live inside the StochasticActor object
print("\n--- 🔍 MODEL SUB-ATTRIBUTES ---")
print(dir(dummy_pop[0].actor))