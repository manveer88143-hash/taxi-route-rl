"""
test.py
-------
Evaluation, demonstration, and multi-model benchmarking.

Covers:
- Phase 2: Random Taxi demonstration & evaluation
- Phase 5: Learned Q-Learning route step-by-step demonstration
- Phase 7: City Taxi with Traffic Avoidance & Passenger pickup route
- FINAL COMPARISON: Benchmark comparison across:
    1. Random Taxi
    2. Shortest-Path Baseline (BFS)
    3. Q-Learning
    4. Deep Q-Network (DQN)
"""

import os
import time
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import SimpleTaxiEnv, CityTaxiEnv
from agent import RandomAgent, QLearningAgent, DQNAgent
from visualization import plot_grid_route, create_route_animation
import config


def run_random_taxi_experiment(num_episodes=100, max_steps=100):
    """
    Phase 2: Evaluates the Random Taxi over multiple episodes.
    Demonstrates why random exploration is inefficient.
    """
    print("\n=======================================================")
    print(f" [PHASE 2] Evaluating Random Taxi ({num_episodes} episodes)")
    print("=======================================================")

    env = SimpleTaxiEnv(grid_size=config.GRID_SIZE,
                        start_pos=config.DEFAULT_START_POS,
                        dest_pos=config.DEFAULT_DEST_POS)
    agent = RandomAgent(env.action_space)

    steps_list = []
    rewards_list = []
    success_list = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps:
            action = agent.choose_action()
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        steps_list.append(steps)
        rewards_list.append(total_reward)
        success_list.append(1 if done else 0)

    avg_steps = float(np.mean(steps_list))
    avg_reward = float(np.mean(rewards_list))
    success_rate = float(np.mean(success_list) * 100)

    print(f"Results for Random Taxi:")
    print(f" - Average Steps:       {avg_steps:.2f} (Max cutoff: {max_steps})")
    print(f" - Average Reward:      {avg_reward:.2f}")
    print(f" - Destination Reached: {success_rate:.1f}%")

    return {
        "model": "Random",
        "avg_steps": round(avg_steps, 2),
        "avg_reward": round(avg_reward, 2),
        "success_rate": round(success_rate, 2)
    }


def demo_learned_route_simple():
    """
    Phase 5: Demonstrates the optimal route learned by the Q-table.
    Shows starting pos, movements, final state, steps, and reward.
    """
    print("\n=======================================================")
    print(" [PHASE 5] Demonstrating Learned Q-Learning Route")
    print("=======================================================")

    env = SimpleTaxiEnv(grid_size=config.GRID_SIZE,
                        start_pos=config.DEFAULT_START_POS,
                        dest_pos=config.DEFAULT_DEST_POS)

    model_path = os.path.join(config.MODELS_DIR, "q_table_simple.npy")
    if not os.path.exists(model_path):
        print(f"[Error] Q-table not found at {model_path}. Run train.py first.")
        return

    agent = QLearningAgent(env.num_states, env.num_actions)
    agent.load(model_path)

    state = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    route_coords = [env.taxi_pos]
    move_symbols = []

    symbol_map = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}

    print("Step-by-step replay using trained Q-table:\n")
    env.render()

    while not done and step_count < 25:
        step_count += 1
        # Greedy action (epsilon = 0, no randomness)
        action = agent.choose_action(state, training=False)
        action_name = env.ACTION_NAMES[action]
        move_symbols.append(symbol_map.get(action_name, "?"))

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        route_coords.append(env.taxi_pos)

        print(f"Step {step_count}: Action {action_name} ({symbol_map.get(action_name)}) -> New Pos {env.taxi_pos}, Reward: {reward}")
        env.render()

    print("Learned Route Path:")
    print(f"START {config.DEFAULT_START_POS} -> " + " -> ".join(move_symbols) + f" -> DEST {config.DEFAULT_DEST_POS}")
    print(f"Total Steps: {step_count} | Total Reward: {total_reward}\n")

    # Plot 2D visual map
    plot_grid_route(
        grid_size=config.GRID_SIZE,
        route=route_coords,
        dest_pos=config.DEFAULT_DEST_POS,
        title="Phase 5: Learned Q-Learning Optimal Route",
        save_path=os.path.join(config.RESULTS_DIR, "q_learning_simple_route.png")
    )

    # Generate Animated GIF
    create_route_animation(
        grid_size=config.GRID_SIZE,
        route=route_coords,
        dest_pos=config.DEFAULT_DEST_POS,
        save_path=os.path.join(config.RESULTS_DIR, "q_learning_simple_route.gif")
    )


def demo_learned_city_route():
    """
    Phases 6-7: Demonstrates passenger pickup and traffic avoidance route.
    """
    print("\n=======================================================")
    print(" [PHASES 6-7] Demonstrating City Route (Passenger + Traffic)")
    print("=======================================================")

    env = CityTaxiEnv(grid_size=config.GRID_SIZE,
                      start_pos=config.DEFAULT_START_POS,
                      passenger_pos=(1, 3),
                      dest_pos=config.DEFAULT_DEST_POS)

    model_path = os.path.join(config.MODELS_DIR, "q_table_city.npy")
    if not os.path.exists(model_path):
        print(f"[Notice] City Q-table not found at {model_path}.")
        return

    agent = QLearningAgent(env.num_states, env.num_actions)
    agent.load(model_path)

    state = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    route_coords = [env.taxi_pos]

    print("Replaying learned City trip:\n")
    env.render()

    while not done and step_count < 30:
        step_count += 1
        action = agent.choose_action(state, training=False)
        action_name = env.ACTION_NAMES[action]

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        route_coords.append(env.taxi_pos)

        print(f"Step {step_count}: {action_name} | Pos: {env.taxi_pos} | InTaxi: {info['passenger_in_taxi']} | Reward: {reward}")
        env.render()

    plot_grid_route(
        grid_size=config.GRID_SIZE,
        route=route_coords,
        traffic_grid=env.traffic_grid,
        passenger_pos=(1, 3),
        dest_pos=config.DEFAULT_DEST_POS,
        title="Phase 7: City Route with Traffic Avoidance & Passenger Pickup",
        save_path=os.path.join(config.RESULTS_DIR, "city_learned_route.png")
    )

    create_route_animation(
        grid_size=config.GRID_SIZE,
        route=route_coords,
        traffic_grid=env.traffic_grid,
        passenger_pos=(1, 3),
        dest_pos=config.DEFAULT_DEST_POS,
        save_path=os.path.join(config.RESULTS_DIR, "city_learned_route.gif")
    )


def run_bfs_shortest_path():
    """
    Baseline: Breadth-First Search (BFS) optimal geometric shortest path.
    Finds minimum distance path on the 5x5 grid without learning.
    """
    start = config.DEFAULT_START_POS
    dest = config.DEFAULT_DEST_POS
    grid_size = config.GRID_SIZE

    queue = deque([[start]])
    visited = set([start])

    while queue:
        path = queue.popleft()
        r, c = path[-1]

        if (r, c) == dest:
            # Reached
            steps = len(path) - 1
            # In simple env: (steps - 1) * -1 + 20
            reward = -(steps - 1) + 20.0
            return {
                "model": "Shortest-Path (BFS)",
                "avg_steps": float(steps),
                "avg_reward": float(reward),
                "success_rate": 100.0
            }

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(path + [(nr, nc)])

    return {"model": "Shortest-Path (BFS)", "avg_steps": 0, "avg_reward": 0, "success_rate": 0.0}


def evaluate_qlearning(num_episodes=100):
    env = SimpleTaxiEnv(grid_size=config.GRID_SIZE,
                        start_pos=config.DEFAULT_START_POS,
                        dest_pos=config.DEFAULT_DEST_POS)
    agent = QLearningAgent(env.num_states, env.num_actions)
    model_path = os.path.join(config.MODELS_DIR, "q_table_simple.npy")
    agent.load(model_path)

    steps_list, rewards_list, success_list = [], [], []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps, total_reward = 0, 0
        while not done and steps < 100:
            action = agent.choose_action(state, training=False)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        steps_list.append(steps)
        rewards_list.append(total_reward)
        success_list.append(1 if done else 0)

    return {
        "model": "Q-Learning",
        "avg_steps": round(float(np.mean(steps_list)), 2),
        "avg_reward": round(float(np.mean(rewards_list)), 2),
        "success_rate": round(float(np.mean(success_list) * 100), 2)
    }


def evaluate_dqn(num_episodes=100):
    if DQNAgent is None:
        return None
    model_path = os.path.join(config.MODELS_DIR, "dqn_taxi.pt")
    if not os.path.exists(model_path):
        return None

    env = SimpleTaxiEnv(grid_size=config.GRID_SIZE,
                        start_pos=config.DEFAULT_START_POS,
                        dest_pos=config.DEFAULT_DEST_POS)
    agent = DQNAgent(env.num_states, env.num_actions)
    agent.load(model_path)

    steps_list, rewards_list, success_list = [], [], []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps, total_reward = 0, 0
        while not done and steps < 100:
            action = agent.choose_action(state, training=False)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        steps_list.append(steps)
        rewards_list.append(total_reward)
        success_list.append(1 if done else 0)

    return {
        "model": "Deep Q-Network (DQN)",
        "avg_steps": round(float(np.mean(steps_list)), 2),
        "avg_reward": round(float(np.mean(rewards_list)), 2),
        "success_rate": round(float(np.mean(success_list) * 100), 2)
    }


def run_full_comparison():
    """
    FINAL COMPARISON:
    Runs all 4 approaches and exports a structured table and bar chart.
    """
    print("\n=======================================================")
    print(" [FINAL BENCHMARK] Comparing All 4 Navigation Methods")
    print("=======================================================")

    results = []
    # 1. Random
    results.append(run_random_taxi_experiment(num_episodes=100))
    # 2. BFS Shortest Path
    results.append(run_bfs_shortest_path())
    # 3. Q-Learning
    results.append(evaluate_qlearning(num_episodes=100))
    # 4. DQN
    dqn_res = evaluate_dqn(num_episodes=100)
    if dqn_res:
        results.append(dqn_res)

    df_comp = pd.DataFrame(results)
    print("\nBenchmark Summary Table:")
    print(df_comp.to_string(index=False))

    # Save CSV
    csv_path = os.path.join(config.RESULTS_DIR, "final_comparison_summary.csv")
    df_comp.to_csv(csv_path, index=False)
    print(f"\n[Saved] Final benchmark CSV saved to: {csv_path}")

    # Generate comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = df_comp["model"]

    # Avg Steps
    axes[0].bar(models, df_comp["avg_steps"], color=["salmon", "skyblue", "mediumseagreen", "mediumpurple"])
    axes[0].set_title("Average Steps (Lower is Better)", fontweight="bold")
    axes[0].set_ylabel("Steps")
    axes[0].tick_params(axis='x', rotation=25)

    # Avg Reward
    axes[1].bar(models, df_comp["avg_reward"], color=["salmon", "skyblue", "mediumseagreen", "mediumpurple"])
    axes[1].set_title("Average Reward (Higher is Better)", fontweight="bold")
    axes[1].set_ylabel("Reward")
    axes[1].tick_params(axis='x', rotation=25)

    # Success Rate
    axes[2].bar(models, df_comp["success_rate"], color=["salmon", "skyblue", "mediumseagreen", "mediumpurple"])
    axes[2].set_title("Success Rate (%)", fontweight="bold")
    axes[2].set_ylabel("Percentage (%)")
    axes[2].set_ylim(0, 110)
    axes[2].tick_params(axis='x', rotation=25)

    plt.tight_layout()
    chart_path = os.path.join(config.RESULTS_DIR, "final_comparison_barchart.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"[Saved] Comparison chart saved to: {chart_path}\n")


if __name__ == "__main__":
    demo_learned_route_simple()
    demo_learned_city_route()
    run_full_comparison()
