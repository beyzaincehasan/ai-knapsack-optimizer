# AI Knapsack Optimizer with Reinforcement Learning

This project compares classical optimization algorithms and a Deep Reinforcement Learning approach for solving the 0/1 Knapsack Problem.

## Project Overview

The implemented methods are:

- Dynamic Programming
- Greedy Algorithm
- Brute Force
- Deep Q-Network (DQN)

The project uses both synthetic data and real benchmark knapsack datasets.

## Algorithms

### Dynamic Programming

Dynamic Programming guarantees the optimal solution but becomes computationally expensive for large capacities.

### Greedy Algorithm

The Greedy algorithm selects items according to their value/weight ratio. It is very fast but does not always guarantee the optimal solution.

### Brute Force

Brute Force checks all possible item combinations. It is used only for small datasets because of exponential complexity.

### Deep Q-Network

DQN is used as a Reinforcement Learning approach. The agent learns whether to select or skip each item based on rewards.

## Project Structure

```text
AlgoritmaAnalizi/
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
├── real_dataset_n_400.png
├── real_dataset_n_1000.png
├── requirements.txt
└── README.md

Technologies
Python
PyTorch
NumPy
Matplotlib
Gymnasium
Installation

Clone the repository:
git clone https://github.com/beyzaincehasan/ai-knapsack-optimizer.git
cd ai-knapsack-optimizer
Create a virtual environment:
python -m venv venv
Activate the virtual environment on Windows:
venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
Running Experiments

Run:
python main.py
Example benchmark datasets:

ks_50_0
ks_100_0
ks_200_0
ks_400_0
ks_1000_0
Experimental Results
Real Benchmark Dataset Results
| Dataset   |    n | Capacity |           DP |  Greedy | DQN Eval Best | Success Rate |
| --------- | ---: | -------: | -----------: | ------: | ------------: | -----------: |
| ks_50_0   |   50 |        - |       142156 |  141956 |             - |       97.85% |
| ks_100_0  |  100 |        - |        99837 |   90000 |             - |       99.66% |
| ks_200_0  |  200 |        - |       100236 |  100062 |             - |       99.27% |
| ks_400_0  |  400 |  9486367 |      3967180 | 3966813 |       3893964 |       98.15% |
| ks_1000_0 | 1000 |   100000 | Not measured |  109869 |        100890 |      91.83%* |
* For ks_1000_0, Dynamic Programming was not measured due to high computational cost. Therefore, the success rate was calculated according to the Greedy result.

Key Findings
Dynamic Programming produces optimal solutions but becomes expensive for large-scale problems.
Brute Force is only feasible for very small datasets.
Greedy is extremely fast but does not guarantee the optimal solution.
DQN achieved near-optimal results on large benchmark datasets.
For ks_400_0, DQN reached 98.15% of the optimal DP result.
For ks_1000_0, DQN reached 91.83% of the Greedy reference result.
Output Graphs

The training process of DQN is saved as graph images, for example:

real_dataset_n_400.png
real_dataset_n_1000.png
Academic Focus

This project demonstrates the use of Reinforcement Learning for combinatorial optimization problems. It compares exact, heuristic, and learning-based approaches in terms of solution quality and scalability.

References
Sutton, R. S., & Barto, A. G. — Reinforcement Learning: An Introduction.
Cormen, T. H. et al. — Introduction to Algorithms.
Mnih, V. et al. — Human-level control through deep reinforcement learning.
Authors
Beyza İncehasan
Gaye Kaymak

Software Engineering Department
Manisa Celal Bayar University