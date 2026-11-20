"""
visualization.py
----------------
Generates clean graphs and visual grid representations for:
- Phase 4: Training curves (Reward, Steps, Success Rate vs Episode)
- Phase 7: Traffic heatmap & optimal route comparison
- Phase 8: Matplotlib grid rendering showing Taxi, Passenger, Traffic, and Learned Route
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_metrics(csv_path, output_image_path):
    """
    Plots the 3 key training curves:
    1. Total Reward vs Episode
    2. Steps vs Episode
    3. Success Rate vs Episode
    """
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Window for rolling average to smooth out exploration noise
    window = 50
    rolling_reward = df["reward"].rolling(window=window, min_periods=1).mean()
    rolling_steps = df["steps"].rolling(window=window, min_periods=1).mean()
    rolling_success = df["success"].rolling(window=window, min_periods=1).mean() * 100

    # 1. Reward Plot
    axes[0].plot(df["episode"], df["reward"], alpha=0.25, color="skyblue", label="Episode Reward")
    axes[0].plot(df["episode"], rolling_reward, color="navy", linewidth=2, label=f"Rolling Avg ({window} eps)")
    axes[0].set_ylabel("Total Reward", fontsize=12, fontweight="bold")
    axes[0].set_title("Reinforcement Learning Training Progress", fontsize=14, fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Steps Plot
    axes[1].plot(df["episode"], df["steps"], alpha=0.25, color="salmon", label="Episode Steps")
    axes[1].plot(df["episode"], rolling_steps, color="darkred", linewidth=2, label=f"Rolling Avg ({window} eps)")
    axes[1].set_ylabel("Steps Taken", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Success Rate Plot
    axes[2].plot(df["episode"], rolling_success, color="green", linewidth=2.5, label="Success Rate (%)")
    axes[2].set_xlabel("Episode Number", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    axes[2].set_ylim(-5, 105)
    axes[2].legend(loc="lower right")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=200)
    plt.close()
    print(f"[Saved] Training progress figure saved to: {output_image_path}")


def plot_grid_route(grid_size, route, traffic_grid=None, passenger_pos=None, dest_pos=None, title="Learned Taxi Route", save_path=None):
    """
    Renders the 5x5 city grid showing:
    - Traffic intensity heatmap
    - Start point
    - Passenger pickup point
    - Destination
    - Trajectory arrows of the taxi's path
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Background traffic
    if traffic_grid is not None:
        cmap = plt.cm.YlOrRd
        im = ax.imshow(traffic_grid, cmap=cmap, vmin=0, vmax=2, alpha=0.45)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(["Low (0.0)", "Medium (-2.0)", "High (-5.0)"])
        cbar.set_label("Traffic Congestion Cost", fontsize=11)
    else:
        ax.imshow(np.zeros((grid_size, grid_size)), cmap="Blues", alpha=0.05)

    # Grid boundaries and ticks
    ax.set_xticks(np.arange(-0.5, grid_size, 1))
    ax.set_yticks(np.arange(-0.5, grid_size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="black", linestyle="-", linewidth=1.5)

    # Label cell coordinates
    for r in range(grid_size):
        for c in range(grid_size):
            traffic_text = ""
            if traffic_grid is not None:
                traffic_text = f"\n(cost {traffic_grid[r, c]})"
            ax.text(c, r, f"({r},{c}){traffic_text}", color="gray", ha="center", va="center", fontsize=8)

    # Plot Route arrows
    if route and len(route) > 1:
        xs = [pt[1] for pt in route]
        ys = [pt[0] for pt in route]
        ax.plot(xs, ys, color="royalblue", linestyle="--", linewidth=2.5, zorder=3)
        for i in range(len(route) - 1):
            r1, c1 = route[i]
            r2, c2 = route[i + 1]
            if (r1, c1) != (r2, c2):
                ax.annotate(
                    "", xy=(c2, r2), xytext=(c1, r1),
                    arrowprops=dict(arrowstyle="->", color="darkblue", lw=3.0, mutation_scale=18),
                    zorder=4
                )

    # Start Point
    start = route[0] if route else (0, 0)
    ax.scatter(start[1], start[0], color="green", s=350, zorder=5, label=f"Taxi Start {start}")
    ax.text(start[1], start[0] - 0.25, "TAXI START", ha="center", fontweight="bold", color="darkgreen", fontsize=9)

    # Passenger Point
    if passenger_pos:
        ax.scatter(passenger_pos[1], passenger_pos[0], color="darkorange", s=350, marker="P", zorder=5, label=f"Passenger {passenger_pos}")
        ax.text(passenger_pos[1], passenger_pos[0] - 0.25, "PASSENGER", ha="center", fontweight="bold", color="chocolate", fontsize=9)

    # Destination Point
    if dest_pos:
        ax.scatter(dest_pos[1], dest_pos[0], color="crimson", s=350, marker="*", zorder=5, label=f"Destination {dest_pos}")
        ax.text(dest_pos[1], dest_pos[0] + 0.32, "DESTINATION", ha="center", fontweight="bold", color="maroon", fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Saved] Grid visualization saved to: {save_path}")
    plt.close()


def create_route_animation(grid_size, route, traffic_grid=None, passenger_pos=None, dest_pos=None, save_path=None):
    """
    Creates an animated GIF showing the taxi smoothly driving along the learned route step-by-step.
    """
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(7, 7))

    def update(frame):
        ax.clear()
        # Background traffic
        if traffic_grid is not None:
            ax.imshow(traffic_grid, cmap=plt.cm.YlOrRd, vmin=0, vmax=2, alpha=0.4)
        else:
            ax.imshow(np.zeros((grid_size, grid_size)), cmap="Blues", alpha=0.05)

        ax.set_xticks(np.arange(-0.5, grid_size, 1))
        ax.set_yticks(np.arange(-0.5, grid_size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color="black", linestyle="-", linewidth=1.5)

        # Label coordinates
        for r in range(grid_size):
            for c in range(grid_size):
                ax.text(c, r, f"({r},{c})", color="gray", ha="center", va="center", fontsize=8)

        # Draw trail up to current frame
        current_path = route[:frame + 1]
        if len(current_path) > 1:
            xs = [pt[1] for pt in current_path]
            ys = [pt[0] for pt in current_path]
            ax.plot(xs, ys, color="royalblue", linestyle="--", linewidth=2.5)

        # Destination marker
        if dest_pos:
            ax.scatter(dest_pos[1], dest_pos[0], color="crimson", s=300, marker="*", zorder=4)
            ax.text(dest_pos[1], dest_pos[0] + 0.3, "DEST", ha="center", fontweight="bold", color="maroon", fontsize=8)

        # Passenger marker
        if passenger_pos:
            # If taxi picked up passenger
            picked_up = frame >= 4  # pickup typically around step 4
            color = "gray" if picked_up else "darkorange"
            ax.scatter(passenger_pos[1], passenger_pos[0], color=color, s=250, marker="P", zorder=4)

        # Current taxi position
        curr_r, curr_c = route[frame]
        ax.scatter(curr_c, curr_r, color="gold", edgecolors="black", s=550, marker="s", zorder=6)
        ax.text(curr_c, curr_r, "TAXI", ha="center", va="center", fontsize=9, fontweight="bold")

        ax.set_title(f"Taxi RL Route Simulation (Step {frame}/{len(route)-1})", fontsize=12, fontweight="bold")

    anim = FuncAnimation(fig, update, frames=len(route), interval=500, repeat_delay=1500)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        anim.save(save_path, writer="pillow", fps=2)
        print(f"[Saved] Route animation GIF saved to: {save_path}")

    plt.close()

