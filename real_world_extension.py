"""
real_world_extension.py
-----------------------
Phase 11: Real-World Road Network Extension using NetworkX.

Demonstrates how to transition from a toy 5x5 grid world to real-world
urban road networks (as seen in OpenStreetMap / Uber / Google Maps):
- Graph representation (Nodes = Intersections with Lat/Lon, Edges = Road Segments)
- Real attributes: road distance (meters), speed limit (km/h), and dynamic traffic congestion index
- Cost calculation: Travel Time = Distance / (Speed * (1 - Congestion))
- Route optimization with shortest-distance vs fastest-route comparison.
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import config


def build_city_road_network():
    """
    Constructs a realistic directed road graph of a city district with 9 key intersections.
    Each edge has:
    - distance: in meters
    - speed_limit: in km/h
    - traffic_factor: 0.0 (empty) to 0.8 (heavy gridlock)
    """
    G = nx.DiGraph()

    # Intersections (nodes) with simulated GPS coordinates (Lat, Lon)
    # Centered around a downtown grid (e.g., Manhattan-style coordinates)
    intersections = {
        "A": {"pos": (-73.985, 40.748), "name": "Grand Central Junction"},
        "B": {"pos": (-73.980, 40.748), "name": "East River Crossing"},
        "C": {"pos": (-73.975, 40.748), "name": "Expressway Inflow"},
        "D": {"pos": (-73.985, 40.742), "name": "Times Square Hub"},
        "E": {"pos": (-73.980, 40.742), "name": "Broadway Downtown (Heavy Traffic)"},
        "F": {"pos": (-73.975, 40.742), "name": "Avenue Bypass"},
        "G": {"pos": (-73.985, 40.736), "name": "West Village Entrance"},
        "H": {"pos": (-73.980, 40.736), "name": "SoHo Transit Point"},
        "I": {"pos": (-73.975, 40.736), "name": "Financial District (Destination)"}
    }

    for node_id, data in intersections.items():
        G.add_node(node_id, pos=data["pos"], name=data["name"])

    # Road Segments (edges): (from_node, to_node, distance_m, speed_kmh, traffic_congestion)
    # Notice: Route through E (Broadway) is short in distance but heavily congested!
    roads = [
        ("A", "B", 400, 30, 0.1),
        ("B", "C", 400, 40, 0.1),
        ("A", "D", 500, 25, 0.2),
        ("B", "E", 500, 20, 0.8),  # Bottleneck!
        ("C", "F", 500, 50, 0.0),  # Fast expressway bypass
        ("D", "E", 400, 15, 0.85), # Severe downtown gridlock
        ("E", "F", 400, 25, 0.6),
        ("D", "G", 500, 30, 0.1),
        ("E", "H", 500, 20, 0.7),
        ("F", "I", 500, 45, 0.05), # Fast arterial link
        ("G", "H", 400, 30, 0.1),
        ("H", "I", 400, 30, 0.2),
    ]

    for u, v, dist, speed, traffic in roads:
        # Effective speed in km/h accounting for congestion slowdown
        effective_speed = max(5.0, speed * (1.0 - traffic))
        # Travel time in seconds: (dist / 1000 km) / (effective_speed km/h) * 3600
        travel_time_sec = (dist / 1000.0) / effective_speed * 3600.0

        # Bidirectional or one-way road attributes
        G.add_edge(u, v, distance=dist, speed=speed, traffic=traffic, travel_time=travel_time_sec)
        G.add_edge(v, u, distance=dist, speed=speed, traffic=traffic, travel_time=travel_time_sec)

    return G


def analyze_real_world_routes():
    """
    Computes and compares:
    1. Shortest Geometric Distance Route (ignores traffic)
    2. Fastest Time-Optimized Route (accounts for real-time traffic delay)
    """
    print("\n=======================================================")
    print(" [PHASE 11] Real-World Network Routing Analysis")
    print("=======================================================")

    G = build_city_road_network()
    origin = "A"        # Grand Central
    destination = "I"   # Financial District

    # 1. Shortest Distance Route (Dijkstra weighted by meters)
    dist_path = nx.shortest_path(G, source=origin, target=destination, weight="distance")
    total_dist = nx.shortest_path_length(G, source=origin, target=destination, weight="distance")
    time_on_dist_path = sum(G[u][v]["travel_time"] for u, v in zip(dist_path[:-1], dist_path[1:]))

    # 2. Fastest Time-Optimized Route (Dijkstra weighted by travel_time)
    fast_path = nx.shortest_path(G, source=origin, target=destination, weight="travel_time")
    fast_time = nx.shortest_path_length(G, source=origin, target=destination, weight="travel_time")
    dist_on_fast_path = sum(G[u][v]["distance"] for u, v in zip(fast_path[:-1], fast_path[1:]))

    print(f"Origin: {origin} ({G.nodes[origin]['name']})")
    print(f"Destination: {destination} ({G.nodes[destination]['name']})\n")

    print(f"1. Shortest Distance Route (Naive GPS):")
    print(f"   Path:        {' -> '.join(dist_path)}")
    print(f"   Distance:    {total_dist:.0f} meters")
    print(f"   Travel Time: {time_on_dist_path / 60.0:.1f} minutes (Trapped in Broadway traffic!)\n")

    print(f"2. Traffic-Aware Fastest Route (AI / Real-Time Navigation):")
    print(f"   Path:        {' -> '.join(fast_path)}")
    print(f"   Distance:    {dist_on_fast_path:.0f} meters (+{dist_on_fast_path - total_dist:.0f} m longer)")
    print(f"   Travel Time: {fast_time / 60.0:.1f} minutes ({((time_on_dist_path - fast_time)/time_on_dist_path)*100:.1f}% faster!)\n")

    # Visualize Network
    fig, ax = plt.subplots(figsize=(10, 7))
    pos = {node: data["pos"] for node, data in G.nodes(data=True)}

    # Draw regular edges
    edge_colors = [G[u][v]["traffic"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", width=1.5, alpha=0.5)

    # Highlight congested roads in red
    congested_edges = [(u, v) for u, v in G.edges() if G[u][v]["traffic"] >= 0.6]
    nx.draw_networkx_edges(G, pos, edgelist=congested_edges, ax=ax, edge_color="red", width=4.0, alpha=0.7, label="Congested Corridors")

    # Highlight optimal fastest path in green
    fast_edges = list(zip(fast_path[:-1], fast_path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=fast_edges, ax=ax, edge_color="limegreen", width=4.5, label="Fastest AI Route")

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightblue", node_size=700, edgecolors="navy")
    labels = {node: f"{node}\n({data['name'][:10]}..)" for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8, font_weight="bold")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="red", lw=3, label="Congested Corridors (Traffic >= 0.6)"),
        Line2D([0], [0], color="limegreen", lw=3.5, label="Fastest AI Route (Avoids Traffic)"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markeredgecolor='navy', markersize=10, label="Intersections")
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    plt.tight_layout()

    save_path = os.path.join(config.RESULTS_DIR, "real_world_network_route.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Saved] Road network visualization saved to: {save_path}\n")


if __name__ == "__main__":
    analyze_real_world_routes()
