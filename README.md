# taxi-route-rl
Taxi route optimization using Reinforcement Learning (PPO) to find efficient routes, reduce travel time, and minimize traffic congestion.
# 🚕 Taxi Route Optimization Using Reinforcement Learning

An intelligent taxi route optimization system using Reinforcement Learning (RL) to find efficient routes in a simulated city road network.

The project uses Proximal Policy Optimization (PPO) to train an intelligent agent that learns to navigate between locations while minimizing travel distance, estimated travel time, and traffic-related costs.

## 🎯 Project Objectives

- Optimize taxi routes using Reinforcement Learning.
- Minimize travel distance and estimated travel time.
- Reduce simulated traffic congestion.
- Train an intelligent routing agent.
- Compare RL performance with traditional pathfinding algorithms.
- Visualize optimized routes and training performance.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| PyTorch | Deep Learning |
| Gymnasium | RL environment |
| Stable-Baselines3 | PPO implementation |
| NetworkX | City road network |
| NumPy | Numerical operations |
| Pandas | Data processing |
| Matplotlib | Visualization |
| Streamlit | Interactive dashboard |

## 🧠 Reinforcement Learning Approach

The project uses Proximal Policy Optimization (PPO).

The agent interacts with a simulated city environment and learns to select suitable roads to reach its destination.

### RL Components

**1. State**

- Current taxi position.
- Destination.
- Available neighboring roads.
- Traffic conditions.
- Remaining distance.

**2. Action**

The agent selects a neighboring intersection to move towards.

**3. Reward Function**

The reward function encourages:
- Reaching the destination.
- Shorter routes.
- Lower travel time.
- Less traffic congestion.

It penalizes:
- Unnecessary movement.
- Invalid actions.
- Repeated visits.
- Excessive travel costs.

## 🏗️ Project Architecture

```text
taxi-route-optimization-rl/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── environment.py
│   ├── city_graph.py
│   ├── agent.py
│   ├── train.py
│   ├── evaluate.py
│   ├── baselines.py
│   └── visualization.py
│
├── results/
│   ├── graphs/
│   └── evaluation/
│
├── app/
│   └── dashboard.py
│
├── tests/
│
├── config/
│   └── config.yaml
│
├── requirements.txt
├── main.py
├── README.md
└── .gitignore
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/taxi-route-optimization-rl.git
```

### 2. Navigate to the project

```bash
cd taxi-route-optimization-rl
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Running the Project

Run the main application:

```bash
python main.py
```

Train the reinforcement learning agent:

```bash
python -m src.train
```

Evaluate the trained model:

```bash
python -m src.evaluate
```

Run the dashboard:

```bash
streamlit run app/dashboard.py
```

Run unit tests:

```bash
pytest tests/
```

*Use these commands after confirming that the corresponding scripts expose these entry points.*

## 📊 Performance Evaluation

The proposed RL model will be evaluated against:

1. Dijkstra's Shortest Path Algorithm.
2. A* Search Algorithm.
3. Random Route Selection.

### Evaluation Metrics

| Metric | Description |
|---|---|
| Total Distance | Distance travelled by taxi |
| Travel Time | Estimated journey duration |
| Route Cost | Combined travel cost |
| Success Rate | Percentage of successful journeys |
| Average Reward | Average RL episode reward |
| Intersections Visited | Number of nodes visited |

Results will be generated through experiments and stored in the `results/` directory.

## 📈 Visualizations

The project generates:

- City road network visualization.
- Optimized taxi route.
- Training reward graph.
- Algorithm comparison charts.
- Travel distance comparison.
- Estimated travel time comparison.

## 🔬 Future Improvements

- Integration with real-world road network data.
- Real-time traffic information.
- GPS-based taxi tracking.
- Multiple taxi optimization.
- Dynamic traffic prediction.
- Integration with mapping APIs.
- Deployment as a real-time route optimization application.

## ⚠️ Limitations

This project initially uses a simulated road network and simulated traffic conditions.

Its performance in a real city requires validation using real-world road, traffic, and travel-time data.

## 👨‍💻 Author

Manveer singh


Machine Learning Project  
Taxi Route Optimization Using Reinforcement Learning
