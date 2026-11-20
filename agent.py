"""
agent.py
--------
Reinforcement Learning Agents for the Taxi Route Optimisation Project.

Contains:
1. RandomAgent (Phase 2): Baseline agent that makes random moves.
2. QLearningAgent (Phases 3-5, 6-7): Tabular Q-Learning built completely from scratch.
3. DQNAgent (Phase 10): Deep Q-Network agent implemented in PyTorch for high-dimensional states.
"""

import random
import numpy as np


class RandomAgent:
    """
    Phase 2: Baseline Random Agent.
    Chooses actions completely at random with no memory or learning.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, state=None):
        return random.choice(self.action_space)


class QLearningAgent:
    """
    Phases 3-5 & 6-7: Tabular Q-Learning Agent implemented from scratch.
    
    Q-Learning Equation:
    Q(s, a) = Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
    """
    def __init__(self, num_states, num_actions, 
                 learning_rate=0.1, 
                 discount_factor=0.95, 
                 epsilon_start=1.0, 
                 epsilon_min=0.01, 
                 epsilon_decay=0.995):
        """
        :param num_states: Total number of discrete states in the environment
        :param num_actions: Number of available actions
        :param learning_rate: Alpha (α) - Step size for Q-value updates (0 < α <= 1)
        :param discount_factor: Gamma (γ) - Value of future rewards (0 <= γ <= 1)
        :param epsilon_start: Initial exploration rate (1.0 = 100% random)
        :param epsilon_min: Lowest value epsilon can decay to
        :param epsilon_decay: Multiplicative decay factor applied after each episode
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # The Q-Table: A 2D matrix of shape (num_states, num_actions), initialized to 0.0
        # Rows = States, Columns = Actions
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float32)

    def choose_action(self, state, training=True):
        """
        Epsilon-Greedy Action Selection:
        - With probability epsilon: EXPLORE (pick random action)
        - With probability 1 - epsilon: EXPLOIT (pick best action according to Q-table)
        """
        if training and (random.random() < self.epsilon):
            # Exploration
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploitation: argmax over actions for this state
            # If multiple actions have the same top Q-value, break ties randomly
            q_values = self.q_table[state]
            max_val = np.max(q_values)
            best_actions = np.where(q_values == max_val)[0]
            return int(np.random.choice(best_actions))

    def update(self, state, action, reward, next_state, done):
        """
        Updates the Q-table entry using the Bellman equation.
        """
        current_q = self.q_table[state, action]
        
        if done:
            # If the episode ended, there are no future actions
            target_q = reward
        else:
            # Best estimated future reward from the next state
            max_future_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_future_q

        # Temporal Difference (TD) error
        td_error = target_q - current_q
        
        # Apply Q-Learning update rule
        self.q_table[state, action] = current_q + self.lr * td_error

    def decay_epsilon(self):
        """Reduces epsilon after each episode so the taxi shifts from exploring to exploiting."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Saves the learned Q-table to disk."""
        np.save(filepath, self.q_table)

    def load(self, filepath):
        """Loads a pre-trained Q-table from disk."""
        self.q_table = np.load(filepath)
        self.epsilon = self.epsilon_min  # Set to evaluation mode


# Try importing PyTorch for DQN (Phase 10)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import deque

    class QNetwork(nn.Module):
        """Feedforward Neural Network approximating Q(s, a)."""
        def __init__(self, state_dim, action_dim):
            super(QNetwork, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )

        def forward(self, x):
            return self.net(x)

    class DQNAgent:
        """
        Phase 10: Deep Q-Network (DQN) Agent using PyTorch.
        Uses Experience Replay and Target Network for stable deep RL.
        """
        def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.95, 
                     epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                     buffer_size=10000, batch_size=32):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.gamma = gamma
            self.epsilon = epsilon_start
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.batch_size = batch_size

            # Primary policy network & Target network
            self.policy_net = QNetwork(state_dim, action_dim)
            self.target_net = QNetwork(state_dim, action_dim)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.criterion = nn.MSELoss()

            # Replay Memory buffer
            self.memory = deque(maxlen=buffer_size)

        def one_hot_state(self, state_int):
            """Encodes discrete integer state into a 1-hot feature vector."""
            vec = np.zeros(self.state_dim, dtype=np.float32)
            vec[state_int] = 1.0
            return torch.FloatTensor(vec).unsqueeze(0)

        def choose_action(self, state_int, training=True):
            if training and (random.random() < self.epsilon):
                return random.randint(0, self.action_dim - 1)
            
            with torch.no_grad():
                state_t = self.one_hot_state(state_int)
                q_values = self.policy_net(state_t)
                return int(torch.argmax(q_values).item())

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

        def train_step(self):
            if len(self.memory) < self.batch_size:
                return

            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            state_vecs = torch.zeros((self.batch_size, self.state_dim))
            next_state_vecs = torch.zeros((self.batch_size, self.state_dim))
            for i in range(self.batch_size):
                state_vecs[i, states[i]] = 1.0
                next_state_vecs[i, next_states[i]] = 1.0

            action_t = torch.LongTensor(actions).unsqueeze(1)
            reward_t = torch.FloatTensor(rewards).unsqueeze(1)
            done_t = torch.FloatTensor(dones).unsqueeze(1)

            # Current Q values
            current_q = self.policy_net(state_vecs).gather(1, action_t)

            # Target Q values using Target Network
            with torch.no_grad():
                next_q = self.target_net(next_state_vecs).max(1)[0].unsqueeze(1)
                target_q = reward_t + (1 - done_t) * self.gamma * next_q

            loss = self.criterion(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def update_target_network(self):
            self.target_net.load_state_dict(self.policy_net.state_dict())

        def decay_epsilon(self):
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        def save(self, filepath):
            torch.save(self.policy_net.state_dict(), filepath)

        def load(self, filepath):
            self.policy_net.load_state_dict(torch.load(filepath, weights_only=True))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon = self.epsilon_min

except ImportError:
    DQNAgent = None
