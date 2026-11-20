# 🚕 Taxi Route Optimisation Using Reinforcement Learning

A beginner-friendly, modular Reinforcement Learning project built from scratch in Python.  
This project demonstrates how an autonomous agent learns to navigate a city grid, pick up passengers, avoid traffic congestion, and reach destinations using **Q-Learning**, **Deep Q-Networks (DQN)**, and **NetworkX Road Graphs**.

---

## 📌 Project Architecture

```text
taxi-route-rl/
├── config.py                 # Central configuration & RL hyperparameters
├── environment.py            # Custom 5x5 Grid environments (SimpleTaxiEnv & CityTaxiEnv)
├── agent.py                  # RandomAgent, QLearningAgent (scratch), DQNAgent (PyTorch)
├── train.py                  # Training pipeline with progress logging & curve plotting
├── test.py                   # Testing, route replays & 4-way benchmark evaluation
├── visualization.py          # 2D heatmaps, trajectory plots & animated GIF generators
├── real_world_extension.py   # Phase 11: NetworkX road graph with real traffic delay costs
├── requirements.txt          # Minimal Python dependencies
├── README.md                 # Project documentation & beginner guide
│
├── data/                     # Route logs and evaluation records
├── models/                   # Saved models: q_table_simple.npy, q_table_city.npy, dqn_taxi.pt
├── results/                  # Training curves, GIFs, and comparison benchmark figures
└── notebooks/                # Jupyter Notebook step-by-step walkthrough
```

---

## 🧠 Core Reinforcement Learning Concepts Explained

### 1. State ($s$)
* The exact situation of the world at any instant.
* In **Simple Taxi (Phases 1–5)**: Taxi position `(row, col)` $\rightarrow$ 25 discrete states.
* In **City Taxi (Phases 6–7)**: `(taxi_row, taxi_col, passenger_in_taxi)` $\rightarrow$ 50 discrete states.

### 2. Action ($a$)
* Decisions the taxi can choose:
  * Simple Grid: `UP` (0), `DOWN` (1), `LEFT` (2), `RIGHT` (3).
  * City Grid: Adds `PICKUP` (4) and `DROPOFF` (5).

### 3. Reward ($r$)
* Feedback returned by the environment:
  * Moving to a standard road: `-1.0` (incentivizes finding the shortest route).
  * Hitting a wall: `-2.0` or `-5.0` penalty.
  * Congested traffic: Additional `-2.0` (medium) or `-5.0` (high) delay penalty.
  * Successful passenger pickup: `+20.0`.
  * Successful dropoff at destination: `+50.0`.

### 4. Q-Table & Q-Values ($Q(s, a)$)
* A table of scores representing: *"If I am in state $s$ and choose action $a$, what total reward will I get in the long run?"*
* Initialized to zeros. As the taxi explores, entries are updated using the Bellman Equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

* **$\alpha$ (Learning Rate = 0.1)**: Determines how quickly new information replaces old knowledge.
* **$\gamma$ (Discount Factor = 0.95)**: Balances immediate rewards vs. long-term future rewards.
* **$\epsilon$ (Epsilon-Greedy)**: Starts at `1.0` (100% random exploration) and decays to `0.01` (99% exploitation of learned optimal actions).

---

## 🚀 How to Run the Project

### Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

### Phase 1: Test Basic Environment
```bash
python environment.py
```

### Phase 2, 5 & Final Benchmark: Evaluate All Models
```bash
python test.py
```
This script runs:
1. **Random Taxi**: Evaluates 100 episodes of random motion to show why exploration alone fails.
2. **Phase 5 Simple Route**: Demonstrates step-by-step navigation using the trained Q-table.
3. **Phases 6–7 City Route**: Demonstrates passenger pickup and traffic avoidance around downtown gridlocks.
4. **Final Comparison**: Compares Random, BFS Shortest-Path, Q-Learning, and Deep Q-Network.

### Phase 3, 4 & 10: Train RL Agents
```bash
python train.py
```
Trains:
* Simple Q-Learning (2,000 episodes, ~0.3s)
* City Taxi Q-Learning with Traffic (3,000 episodes, ~0.7s)
* Deep Q-Network in PyTorch (800 episodes, ~21s)

### Phase 11: Real-World Road Network (NetworkX)
```bash
python real_world_extension.py
```
Demonstrates routing on a real graph where the shortest distance route (Broadway) is gridlocked, and AI routes traffic around the arterial expressways to save 77.8% travel time.

---

## 📊 Final Benchmark Comparison

| Method | Average Steps | Average Reward | Destination Success Rate |
| :--- | :---: | :---: | :---: |
| **Random Taxi** | 73.29 | -75.79 | 60.0% |
| **Shortest-Path (BFS)** | 8.00 | +13.00 | 100.0% |
| **Q-Learning (Scratch)** | **8.00** | **+13.00** | **100.0%** |
| **Deep Q-Network (DQN)** | **8.00** | **+13.00** | **100.0%** |

*Note: On a 5×5 grid with start (0,0) and destination (4,4), the theoretical minimum Manhattan distance is 8 steps $((4-0) + (4-0) = 8)$. Both Q-Learning and DQN achieved 100% mathematical optimality!*
