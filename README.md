AI Knapsack Optimizer with Reinforcement Learning

A comparative study of classical optimization algorithms and Deep Reinforcement Learning (DQN) for solving the Knapsack Problem using both synthetic and real benchmark datasets.

📌 Project Overview

This project focuses on solving the classical 0/1 Knapsack Problem using multiple approaches:

Dynamic Programming (DP)
Greedy Algorithm
Brute Force
Deep Q-Network (DQN)

The main objective is to compare classical optimization techniques with a Reinforcement Learning-based approach and analyze their performance on different problem scales.

The project includes:

synthetic data experiments
real benchmark knapsack datasets
large-scale experiments (n = 1000)
learning performance visualization
scalability analysis
🧠 Algorithms Used
1. Dynamic Programming (DP)

Provides optimal solutions but becomes computationally expensive for large-scale problems.

2. Greedy Algorithm

Very fast heuristic approach based on value/weight ratio.

3. Brute Force

Checks all possible combinations.
Used only for very small datasets due to exponential complexity.

4. Deep Q-Network (DQN)

A Reinforcement Learning approach where an agent learns to maximize total knapsack value through interaction with the environment.

Features:

experience replay
target network
epsilon-greedy exploration
reward-based learning
📂 Project Structure
AlgoritmaAnalizi/
│
├── main.py
├── experiment.py
├── knapsack_algorithms.py
├── knapsack_env.py
├── knapsack_dqn.py
├── knapsack_data/
│   └── data/
│       ├── ks_50_0
│       ├── ks_100_0
│       ├── ks_200_0
│       ├── ks_400_0
│       └── ks_1000_0
│
├── plots/
├── requirements.txt
└── README.md
⚙️ Technologies
Python
PyTorch
NumPy
Matplotlib
Reinforcement Learning (DQN)
🚀 Installation

Clone the repository:

git clone https://github.com/beyzaincehasan/ai-knapsack-optimizer.git
cd ai-knapsack-optimizer

Create virtual environment:

python -m venv venv

Activate environment:

Windows
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
▶️ Running Experiments

Run benchmark dataset experiments:

python main.py

Example datasets:

ks_50_0
ks_100_0
ks_200_0
ks_400_0
ks_1000_0
📊 Experimental Results
Real Dataset Results
n	DP	Greedy	DQN Eval Best	Success Rate
50	142156	141956	97.85%	
100	99837	90000	99.66%	
200	100236	100062	99.27%	
400	3967180	3966813	98.15%	
1000	Not Measured	109869	91.83% (Greedy Reference)	
📈 Key Findings
Dynamic Programming guarantees optimal solutions but becomes computationally expensive at large scales.
Greedy algorithm is extremely fast but may fail to find optimal solutions.
DQN achieved near-optimal performance on large datasets.
On n = 400, DQN reached approximately 98.15% of the optimal solution.
On n = 1000, where DP could not be measured due to computational limitations, DQN still achieved 91.83% of the Greedy reference solution.
📚 Academic Focus

This project demonstrates how Reinforcement Learning can be used in combinatorial optimization problems and highlights scalability advantages over classical exact methods.

📖 References
Sutton & Barto — Reinforcement Learning: An Introduction
Cormen et al. — Introduction to Algorithms
Mnih et al. — Human-level control through deep reinforcement learning
👩‍💻 Authors
Beyza İncehasan
Gaye Kaymak

Software Engineering Department
Manisa Celal Bayar University