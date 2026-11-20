"""
environment.py
--------------
Reinforcement Learning Taxi Environments built from scratch.

Contains two modular environments:
1. SimpleTaxiEnv (Phases 1-5):
   - 5x5 Grid World
   - 4 Actions: UP, DOWN, LEFT, RIGHT
   - Taxi navigates to destination.

2. CityTaxiEnv (Phases 6-7):
   - 5x5 Grid World with Passenger & Traffic
   - 6 Actions: UP, DOWN, LEFT, RIGHT, PICKUP, DROPOFF
   - Taxi must find passenger, pick up, navigate through traffic, and drop off at destination.
"""

import sys
import numpy as np

# Ensure UTF-8 output if possible, fallback safely on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class SimpleTaxiEnv:
    """
    Phase 1-5: Basic 5x5 grid environment for navigation.
    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    """
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    
    ACTION_NAMES = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT"
    }

    def __init__(self, grid_size=5, start_pos=(0, 0), dest_pos=(4, 4)):
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.dest_pos = dest_pos
        self.taxi_pos = start_pos
        
        self.action_space = [
            self.ACTION_UP, 
            self.ACTION_DOWN, 
            self.ACTION_LEFT, 
            self.ACTION_RIGHT
        ]
        self.num_actions = len(self.action_space)
        self.num_states = self.grid_size * self.grid_size

    def encode_state(self, pos=None):
        """Converts (row, col) into a single unique integer from 0 to 24."""
        if pos is None:
            pos = self.taxi_pos
        r, c = pos
        return r * self.grid_size + c

    def decode_state(self, state_int):
        """Converts an integer state back into (row, col)."""
        r = state_int // self.grid_size
        c = state_int % self.grid_size
        return (r, c)

    def reset(self, start_pos=None):
        """Resets the environment for a new episode."""
        if start_pos is not None:
            self.taxi_pos = start_pos
        else:
            self.taxi_pos = self.start_pos
        return self.encode_state()

    def get_state(self):
        return self.encode_state()

    def step(self, action):
        """
        Executes one movement.
        Returns: (next_state_int, reward, done, info)
        """
        row, col = self.taxi_pos
        
        if action == self.ACTION_UP:
            new_row, new_col = row - 1, col
        elif action == self.ACTION_DOWN:
            new_row, new_col = row + 1, col
        elif action == self.ACTION_LEFT:
            new_row, new_col = row, col - 1
        elif action == self.ACTION_RIGHT:
            new_row, new_col = row, col + 1
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check grid boundaries
        if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
            self.taxi_pos = (new_row, new_col)
            hit_wall = False
        else:
            hit_wall = True

        done = (self.taxi_pos == self.dest_pos)

        # Rewards:
        # Reached destination: +20
        # Normal step: -1
        # Hitting a wall: -2 (discourages ramming walls)
        if done:
            reward = 20.0
        elif hit_wall:
            reward = -2.0
        else:
            reward = -1.0

        next_state = self.encode_state()
        info = {
            "pos": self.taxi_pos,
            "action_name": self.ACTION_NAMES[action],
            "hit_wall": hit_wall,
            "done": done
        }
        return next_state, reward, done, info

    def render(self):
        """Renders ASCII representation of 5x5 grid."""
        s_both = " * "
        s_taxi = " T "
        s_dest = " G "
        s_empty = " . "

        cell_len = len(s_empty)
        print("+" + "-" * (self.grid_size * cell_len) + "+")
        for r in range(self.grid_size):
            row_display = []
            for c in range(self.grid_size):
                if (r, c) == self.taxi_pos and (r, c) == self.dest_pos:
                    row_display.append(s_both)
                elif (r, c) == self.taxi_pos:
                    row_display.append(s_taxi)
                elif (r, c) == self.dest_pos:
                    row_display.append(s_dest)
                else:
                    row_display.append(s_empty)
            print("|" + "".join(row_display) + "|")
        print("+" + "-" * (self.grid_size * cell_len) + "+")
        print(f"Taxi: {self.taxi_pos} | Destination: {self.dest_pos}\n")


class CityTaxiEnv:
    """
    Phases 6-7: Extended City Environment with:
    - Passenger pickup and dropoff
    - Variable traffic conditions
    6 Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=PICKUP, 5=DROPOFF
    """
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

    def __init__(self, grid_size=5, start_pos=(0, 0), passenger_pos=(1, 3), dest_pos=(4, 4), traffic_grid=None):
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.passenger_pos = passenger_pos
        self.dest_pos = dest_pos
        
        # State tracking
        self.taxi_pos = start_pos
        self.passenger_in_taxi = False
        
        # 6 Discrete actions
        self.action_space = [0, 1, 2, 3, 4, 5]
        self.num_actions = len(self.action_space)

        # Traffic grid: 0 = Low (0.0 penalty), 1 = Med (-2.0 penalty), 2 = High (-5.0 penalty)
        if traffic_grid is not None:
            self.traffic_grid = traffic_grid
        else:
            # Default realistic city traffic pattern
            # High traffic in the center / bottleneck cells
            self.traffic_grid = np.zeros((grid_size, grid_size), dtype=int)
            # Create a congested downtown stripe
            self.traffic_grid[2, 1:4] = 2  # HIGH traffic
            self.traffic_grid[1, 2] = 1    # MEDIUM traffic
            self.traffic_grid[3, 2] = 1    # MEDIUM traffic

        self.traffic_penalties = {
            0: 0.0,   # Low
            1: -2.0,  # Medium
            2: -5.0   # High
        }

        # Number of states: 25 taxi cells * 2 passenger states (0: at pickup, 1: in taxi) = 50 states
        self.num_states = self.grid_size * self.grid_size * 2

    def encode_state(self):
        """
        Encodes (taxi_row, taxi_col, passenger_in_taxi) into a single integer.
        Returns integer in range [0, 49].
        """
        r, c = self.taxi_pos
        pos_id = r * self.grid_size + c
        p_status = 1 if self.passenger_in_taxi else 0
        return pos_id * 2 + p_status

    def decode_state(self, state_int):
        """Decodes integer back to (taxi_row, taxi_col, passenger_in_taxi)."""
        p_status = bool(state_int % 2)
        pos_id = state_int // 2
        r = pos_id // self.grid_size
        c = pos_id % self.grid_size
        return (r, c, p_status)

    def reset(self, start_pos=None):
        if start_pos is not None:
            self.taxi_pos = start_pos
        else:
            self.taxi_pos = self.start_pos
        self.passenger_in_taxi = False
        return self.encode_state()

    def step(self, action):
        """
        Executes one action (movement, pickup, or dropoff).
        Returns: (next_state, reward, done, info)
        """
        row, col = self.taxi_pos
        reward = 0.0
        done = False
        info = {
            "action_name": self.ACTION_NAMES[action],
            "taxi_pos": self.taxi_pos,
            "passenger_in_taxi": self.passenger_in_taxi,
            "traffic_level": "LOW"
        }

        # --- MOVEMENT ACTIONS ---
        if action in [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]:
            if action == self.ACTION_UP:
                new_row, new_col = row - 1, col
            elif action == self.ACTION_DOWN:
                new_row, new_col = row + 1, col
            elif action == self.ACTION_LEFT:
                new_row, new_col = row, col - 1
            elif action == self.ACTION_RIGHT:
                new_row, new_col = row, col + 1

            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                self.taxi_pos = (new_row, new_col)
                # Traffic penalty in new cell
                traffic_val = self.traffic_grid[new_row, new_col]
                traffic_cost = self.traffic_penalties[traffic_val]
                # Base step cost (-1) + traffic delay cost
                reward = -1.0 + traffic_cost
                level_names = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
                info["traffic_level"] = level_names.get(traffic_val, "LOW")
            else:
                # Wall hit
                reward = -5.0

        # --- PICKUP ACTION ---
        elif action == self.ACTION_PICKUP:
            if not self.passenger_in_taxi and self.taxi_pos == self.passenger_pos:
                self.passenger_in_taxi = True
                reward = 20.0  # Successful pickup
            else:
                reward = -10.0  # Illegal pickup attempt

        # --- DROPOFF ACTION ---
        elif action == self.ACTION_DROPOFF:
            if self.passenger_in_taxi and self.taxi_pos == self.dest_pos:
                self.passenger_in_taxi = False
                done = True
                reward = 50.0  # Successful mission completion!
            else:
                reward = -10.0  # Illegal dropoff attempt
        else:
            raise ValueError(f"Invalid action: {action}")

        next_state = self.encode_state()
        info["taxi_pos"] = self.taxi_pos
        info["passenger_in_taxi"] = self.passenger_in_taxi
        info["done"] = done

        return next_state, reward, done, info

    def render(self):
        """Renders ASCII representation with Passenger (P), Destination (G), and Traffic."""
        traffic_icons = {0: " ", 1: "~", 2: "#"}  # Normal, Medium, High traffic indicator
        print("+" + "----" * self.grid_size + "+")
        for r in range(self.grid_size):
            row_str = []
            for c in range(self.grid_size):
                t_symbol = traffic_icons[self.traffic_grid[r, c]]
                if (r, c) == self.taxi_pos:
                    if self.passenger_in_taxi:
                        cell = f"T+P{t_symbol}"
                    else:
                        cell = f" T {t_symbol}"
                elif not self.passenger_in_taxi and (r, c) == self.passenger_pos:
                    cell = f" P {t_symbol}"
                elif (r, c) == self.dest_pos:
                    cell = f" G {t_symbol}"
                else:
                    cell = f" . {t_symbol}"
                row_str.append(cell)
            print("|" + "".join(row_str) + "|")
        print("+" + "----" * self.grid_size + "+")
        p_status = "In Taxi" if self.passenger_in_taxi else f"At {self.passenger_pos}"
        print(f"Taxi: {self.taxi_pos} | Passenger: {p_status} | Dest: {self.dest_pos}")
        print("Traffic legend: [ ] Low traffic, [~] Medium delay, [#] High congestion\n")
