"""
train.py
--------
Training scripts for Taxi Route Optimisation.

Supports:
1. Training Q-Learning on SimpleTaxiEnv (Phases 3-4)
2. Training Q-Learning on CityTaxiEnv with Passenger & Traffic (Phases 6-7)
3. Training Deep Q-Network (DQN) on SimpleTaxiEnv (Phase 10)
"""

import os
import time
import pandas as pd
import numpy as np

from environment import SimpleTaxiEnv, CityTaxiEnv
from agent import QLearningAgent, DQNAgent
from visualization import plot_training_metrics
import config


def train_qlearning_simple(num_episodes=2000, max_steps=100):
    """
    Trains tabular Q-Learning on the 5x5 SimpleTaxiEnv.
    """
    print(f"\n=======================================================")
    print(f" [TRAIN] Q-Learning on Simple 5x5 Taxi ({num_episodes} episodes)")
    print(f"=======================================================")

    env = SimpleTaxiEnv(grid_size=config.GRID_SIZE, 
                        start_pos=config.DEFAULT_START_POS, 
                        dest_pos=config.DEFAULT_DEST_POS)

    agent = QLearningAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        learning_rate=config.QL_LEARNING_RATE,
        discount_factor=config.QL_DISCOUNT_FACTOR,
        epsilon_start=config.QL_EPSILON_START,
        epsilon_min=config.QL_EPSILON_MIN,
        epsilon_decay=config.QL_EPSILON_DECAY
    )

    history = []
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            # Bellman Q-Learning update
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        success = 1 if done else 0
        history.append({
            "episode": episode,
            "reward": total_reward,
            "steps": steps,
            "success": success,
            "epsilon": round(agent.epsilon, 4)
        })

        if episode % 200 == 0 or episode == num_episodes:
            recent_success = np.mean([h["success"] for h in history[-100:]]) * 100
            recent_reward = np.mean([h["reward"] for h in history[-100:]])
            print(f"Episode {episode:4d}/{num_episodes} | Avg Reward: {recent_reward:6.2f} | "
                  f"Success Rate: {recent_success:5.1f}% | Epsilon: {agent.epsilon:.3f}")

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds.")

    # Save Q-table
    model_path = os.path.join(config.MODELS_DIR, "q_table_simple.npy")
    agent.save(model_path)
    print(f"[Saved] Q-table saved to: {model_path}")

    # Save metrics CSV
    metrics_path = os.path.join(config.RESULTS_DIR, "q_learning_simple_metrics.csv")
    df = pd.DataFrame(history)
    df.to_csv(metrics_path, index=False)
    print(f"[Saved] Metrics saved to: {metrics_path}")

    # Generate graphs
    plot_path = os.path.join(config.RESULTS_DIR, "q_learning_training_curves.png")
    plot_training_metrics(metrics_path, plot_path)

    return agent, df


def train_qlearning_city(num_episodes=3000, max_steps=120):
    """
    Trains tabular Q-Learning on the CityTaxiEnv (with passenger & traffic).
    """
    print(f"\n=======================================================")
    print(f" [TRAIN] Q-Learning on City Taxi + Traffic ({num_episodes} episodes)")
    print(f"=======================================================")

    env = CityTaxiEnv(grid_size=config.GRID_SIZE,
                      start_pos=config.DEFAULT_START_POS,
                      passenger_pos=(1, 3),
                      dest_pos=config.DEFAULT_DEST_POS)

    agent = QLearningAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        learning_rate=config.QL_LEARNING_RATE,
        discount_factor=config.QL_DISCOUNT_FACTOR,
        epsilon_start=config.QL_EPSILON_START,
        epsilon_min=config.QL_EPSILON_MIN,
        epsilon_decay=0.997  # Slightly slower decay to thoroughly explore pickup
    )

    history = []
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        success = 1 if done else 0
        history.append({
            "episode": episode,
            "reward": total_reward,
            "steps": steps,
            "success": success,
            "epsilon": round(agent.epsilon, 4)
        })

        if episode % 500 == 0 or episode == num_episodes:
            recent_success = np.mean([h["success"] for h in history[-100:]]) * 100
            recent_reward = np.mean([h["reward"] for h in history[-100:]])
            print(f"Episode {episode:4d}/{num_episodes} | Avg Reward: {recent_reward:6.2f} | "
                  f"Success Rate: {recent_success:5.1f}% | Epsilon: {agent.epsilon:.3f}")

    elapsed = time.time() - start_time
    print(f"City Training completed in {elapsed:.2f} seconds.")

    model_path = os.path.join(config.MODELS_DIR, "q_table_city.npy")
    agent.save(model_path)
    print(f"[Saved] City Q-table saved to: {model_path}")

    metrics_path = os.path.join(config.RESULTS_DIR, "q_learning_city_metrics.csv")
    df = pd.DataFrame(history)
    df.to_csv(metrics_path, index=False)
    print(f"[Saved] City Metrics saved to: {metrics_path}")

    plot_path = os.path.join(config.RESULTS_DIR, "city_training_curves.png")
    plot_training_metrics(metrics_path, plot_path)

    return agent, df


def train_dqn(num_episodes=800, max_steps=100):
    """
    Trains Deep Q-Network (PyTorch) on the SimpleTaxiEnv.
    """
    if DQNAgent is None:
        print("[Warning] PyTorch is not available; skipping DQN training.")
        return None, None

    print(f"\n=======================================================")
    print(f" [TRAIN] Deep Q-Network (DQN) on Simple Taxi ({num_episodes} episodes)")
    print(f"=======================================================")

    env = SimpleTaxiEnv(grid_size=config.GRID_SIZE,
                        start_pos=config.DEFAULT_START_POS,
                        dest_pos=config.DEFAULT_DEST_POS)

    agent = DQNAgent(
        state_dim=env.num_states,
        action_dim=env.num_actions,
        lr=0.001,
        gamma=config.QL_DISCOUNT_FACTOR,
        epsilon_start=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.993,
        batch_size=32
    )

    history = []
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            # Store transition in replay memory
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward
            steps += 1

        # Periodically update target network weights
        if episode % 10 == 0:
            agent.update_target_network()

        agent.decay_epsilon()
        success = 1 if done else 0
        history.append({
            "episode": episode,
            "reward": total_reward,
            "steps": steps,
            "success": success,
            "epsilon": round(agent.epsilon, 4)
        })

        if episode % 100 == 0 or episode == num_episodes:
            recent_success = np.mean([h["success"] for h in history[-50:]]) * 100
            recent_reward = np.mean([h["reward"] for h in history[-50:]])
            print(f"Episode {episode:4d}/{num_episodes} | Avg Reward: {recent_reward:6.2f} | "
                  f"Success Rate: {recent_success:5.1f}% | Epsilon: {agent.epsilon:.3f}")

    elapsed = time.time() - start_time
    print(f"DQN Training completed in {elapsed:.2f} seconds.")

    model_path = os.path.join(config.MODELS_DIR, "dqn_taxi.pt")
    agent.save(model_path)
    print(f"[Saved] DQN Model saved to: {model_path}")

    metrics_path = os.path.join(config.RESULTS_DIR, "dqn_metrics.csv")
    df = pd.DataFrame(history)
    df.to_csv(metrics_path, index=False)

    plot_path = os.path.join(config.RESULTS_DIR, "dqn_training_curves.png")
    plot_training_metrics(metrics_path, plot_path)

    return agent, df


if __name__ == "__main__":
    train_qlearning_simple()
    train_qlearning_city()
    train_dqn()
