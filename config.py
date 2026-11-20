"""
config.py
---------
Configuration parameters and hyperparameters for the Taxi RL project.
All parameters are centralized here so they can be easily tuned and referenced.
"""

import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Output directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

# Ensure folders exist
for folder in [DATA_DIR, MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Grid World Settings
GRID_SIZE = 5
DEFAULT_START_POS = (0, 0)
DEFAULT_DEST_POS = (4, 4)

# Available Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_PICKUP = 4
ACTION_DROPOFF = 5

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "PICKUP",
    5: "DROPOFF"
}

# Reward Structure (Basic & Advanced)
STEP_PENALTY = -1.0
INVALID_MOVE_PENALTY = -5.0
SUCCESS_REWARD = 50.0
PASSENGER_PICKUP_REWARD = 20.0
WRONG_ACTION_PENALTY = -10.0

# Traffic Costs (Phase 7)
TRAFFIC_PENALTIES = {
    "LOW": 0.0,       # Normal road, standard step penalty only
    "MEDIUM": -2.0,   # Moderate delay
    "HIGH": -5.0      # Heavy congestion
}

# Q-Learning Hyperparameters (Phase 3 & 4)
QL_LEARNING_RATE = 0.1       # Alpha (α): how fast the Q-values update
QL_DISCOUNT_FACTOR = 0.95    # Gamma (γ): importance of future rewards vs immediate
QL_EPSILON_START = 1.0       # Initial exploration rate (100% random at start)
QL_EPSILON_MIN = 0.01        # Minimum exploration rate (agent keeps 1% randomness)
QL_EPSILON_DECAY = 0.995     # Decay multiplier per episode
QL_TRAIN_EPISODES = 2000     # Total training episodes
MAX_STEPS_PER_EPISODE = 100  # Cutoff to avoid infinite loops during early exploration

# Deep Q-Network (DQN) Hyperparameters (Phase 10)
DQN_BATCH_SIZE = 32
DQN_REPLAY_BUFFER_SIZE = 10000
DQN_LEARNING_RATE = 0.001
DQN_GAMMA = 0.95
DQN_TARGET_UPDATE_FREQ = 10
DQN_TRAIN_EPISODES = 1000
