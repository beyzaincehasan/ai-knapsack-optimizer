import random
import time
from typing import List, Dict, Tuple


def set_seed(seed: int = 42):
    random.seed(seed)


import random
from typing import List, Dict

def generate_network_traffic(n: int = 10, seed: int = 42) -> List[Dict]:
    """
    Knapsack verisini daha zor ve gerçekçi şekilde üretir.

    weight -> bant genişliği ihtiyacı
    value  -> öncelik/değer

    - Bazı item'lar kötü (low value / high weight)
    - Bazı item'lar çok iyi (high value / low weight)
    - Greedy'nin hata yapabileceği senaryolar oluşturulur
    """
    random.seed(seed)
    traffic = []

    for i in range(n):
        r = random.random()

        # %30 kötü item (greedy'yi yanıltır)
        if r < 0.3:
            weight = random.randint(15, 30)
            value = random.randint(10, 40)

        # %50 normal item
        elif r < 0.8:
            weight = random.randint(5, 20)
            value = random.randint(40, 90)

        # %20 çok iyi item (yüksek değerli)
        else:
            weight = random.randint(1, 10)
            value = random.randint(80, 150)

        packet = {
            "id": i,
            "weight": weight,
            "value": value
        }

        traffic.append(packet)

    return traffic


def extract_weights_values(traffic: List[Dict]) -> Tuple[List[int], List[int]]:
    weights = [item["weight"] for item in traffic]
    values = [item["value"] for item in traffic]
    return weights, values


def knapsack_brute_force(W: int, weights: List[int], values: List[int], n: int) -> int:
    if n == 0 or W == 0:
        return 0
    if weights[n - 1] > W:
        return knapsack_brute_force(W, weights, values, n - 1)

    return max(
        values[n - 1] + knapsack_brute_force(W - weights[n - 1], weights, values, n - 1),
        knapsack_brute_force(W, weights, values, n - 1)
    )


def knapsack_greedy(W: int, traffic: List[Dict]) -> int:
    sorted_traffic = sorted(traffic, key=lambda x: x["value"] / x["weight"], reverse=True)
    total_value = 0
    current_weight = 0

    for packet in sorted_traffic:
        if current_weight + packet["weight"] <= W:
            current_weight += packet["weight"]
            total_value += packet["value"]

    return total_value


def knapsack_dp(W: int, weights: List[int], values: List[int], n: int) -> int:
    K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif weights[i - 1] <= w:
                K[i][w] = max(
                    values[i - 1] + K[i - 1][w - weights[i - 1]],
                    K[i - 1][w]
                )
            else:
                K[i][w] = K[i - 1][w]

    return K[n][W]


def measure_time(func, *args):
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return result, (end - start) * 1000

def load_knapsack_file(file_path: str):
    """
    Gerçek knapsack veri dosyasını okur.

    Dosya formatı:
    İlk satır: n capacity
    Sonraki satırlar: value weight
    """
    traffic = []

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    first_line = lines[0].strip().split()
    n = int(first_line[0])
    capacity = int(first_line[1])

    for i, line in enumerate(lines[1:n + 1]):
        value, weight = map(int, line.strip().split())

        packet = {
            "id": i,
            "value": value,
            "weight": weight
        }

        traffic.append(packet)

    return traffic, capacity