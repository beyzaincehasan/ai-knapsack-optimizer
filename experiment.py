import matplotlib.pyplot as plt

from knapsack_algorithms import (
    generate_network_traffic,
    load_knapsack_file,
    extract_weights_values,
    knapsack_brute_force,
    knapsack_greedy,
    knapsack_dp,
    measure_time
)
from knapsack_env import KnapsackEnv
from knapsack_dqn import DQNAgent


def train_dqn(traffic_data, capacity, episodes=300, batch_size=64):
    env = KnapsackEnv(traffic_data, capacity)
    agent = DQNAgent(state_size=5, action_size=2)

    scores = []

    for episode in range(episodes):
        state, _ = env.reset()

        for _ in range(len(traffic_data)):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                scores.append(env.total_value)
                break

        if (episode + 1) % 20 == 0:
            agent.update_target_model()

    agent.epsilon = 0.0
    eval_scores = []

    for _ in range(20):
        state, _ = env.reset()

        for _ in range(len(traffic_data)):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state

            if done:
                eval_scores.append(env.total_value)
                break

    best_eval = max(eval_scores)
    avg_eval = sum(eval_scores) / len(eval_scores)

    return scores, best_eval, avg_eval


def compare_algorithms(n=15, capacity=50, seed=42, episodes=300):
    traffic = generate_network_traffic(n=n, seed=seed)
    return run_comparison(traffic, capacity, episodes, title=f"DQN Öğrenme Süreci (n={n})", filename=f"plot_n_{n}.png")


def compare_algorithms_from_file(file_path, episodes=1000):
    traffic, capacity = load_knapsack_file(file_path)
    n = len(traffic)

    return run_comparison(
        traffic,
        capacity,
        episodes,
        title=f"DQN Öğrenme Süreci - Gerçek Veri (n={n})",
        filename=f"real_dataset_n_{n}.png",
        file_path=file_path
    )


def run_comparison(traffic, capacity, episodes, title, filename, file_path=None):
    n = len(traffic)
    weights, values = extract_weights_values(traffic)
    if n <= 400:
     dp_result, dp_time = measure_time(
        knapsack_dp, capacity, weights, values, n
    )
    else:
     dp_result, dp_time = None, None
    
    greedy_result, greedy_time = measure_time(knapsack_greedy, capacity, traffic)

    if n <= 18:
        bf_result, bf_time = measure_time(
            knapsack_brute_force, capacity, weights, values, n
        )
    else:
        bf_result, bf_time = None, None

    dqn_scores, dqn_best_eval, dqn_avg_eval = train_dqn(
        traffic, capacity, episodes=episodes
    )

    dqn_final = dqn_scores[-1]
    dqn_best = max(dqn_scores)

    if file_path:
        print("\n=== GERÇEK VERİ SETİ SONUÇLARI ===")
        print(f"Dosya         : {file_path}")
    else:
        print("\n=== SONUÇLAR ===")

    print(f"n             : {n}")
    print(f"Capacity      : {capacity}")
    if dp_result is not None:
      print(f"DP            : {dp_result} | {dp_time:.4f} ms")
    else:
     print("DP            : problem boyutu çok büyük olduğu için ölçülmedi")
    print(f"Greedy        : {greedy_result} | {greedy_time:.4f} ms")

    if bf_result is not None:
        print(f"BruteForce    : {bf_result} | {bf_time:.4f} ms")
    else:
        print("BruteForce    : büyük n nedeniyle ölçülmedi")

    print(f"DQN Final     : {dqn_final}")
    print(f"DQN Best      : {dqn_best}")
    print(f"DQN Eval Best : {dqn_best_eval}")
    print(f"DQN Eval Avg  : {dqn_avg_eval:.2f}")
    if dp_result is not None:
     print(f"DQN Eval Best / DP Başarı: %{(dqn_best_eval / dp_result) * 100:.2f}")
    else:
     print("DQN Eval Best / DP Başarı: DP ölçülmediği için hesaplanamadı")
    if greedy_result is not None:
     print(f"DQN Eval Best / Greedy Başarı: %{(dqn_best_eval / greedy_result) * 100:.2f}")
    plt.figure(figsize=(10, 5))
    plt.plot(dqn_scores, label="DQN Eğitim Skoru")
    if dp_result is not None:
     plt.axhline(y=dp_result, linestyle="--", label="DP Optimal")
    else:
     plt.axhline(y=greedy_result, linestyle="--", label="Greedy Reference")
    plt.axhline(y=greedy_result, linestyle=":", label="Greedy")
    plt.xlabel("Episode")
    plt.ylabel("Toplam Değer")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()

    return {
        "n": n,
        "capacity": capacity,
        "dp_result": dp_result,
        "dp_time": dp_time,
        "greedy_result": greedy_result,
        "greedy_time": greedy_time,
        "bf_result": bf_result,
        "bf_time": bf_time,
        "dqn_final": dqn_final,
        "dqn_best": dqn_best,
        "dqn_best_eval": dqn_best_eval,
        "dqn_avg_eval": dqn_avg_eval,
        "dqn_scores": dqn_scores
    }