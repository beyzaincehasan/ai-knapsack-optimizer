import gymnasium as gym
from gymnasium import spaces
import numpy as np


class KnapsackEnv(gym.Env):
    def __init__(self, traffic_data, capacity):
        super(KnapsackEnv, self).__init__()
        self.traffic_data = traffic_data
        self.capacity = capacity
        self.n = len(traffic_data)

        self.action_space = spaces.Discrete(2)

        # State:
        # [current_weight, remaining_capacity, item_weight, item_value, remaining_items]
        self.observation_space = spaces.Box(
            low=0,
            high=max(capacity, 1000),
            shape=(5,),
            dtype=np.float32
        )

        self.reset()

    def _get_state(self):
        if self.current_packet_idx >= self.n:
            return np.zeros(5, dtype=np.float32)

        packet = self.traffic_data[self.current_packet_idx]
        remaining_capacity = self.capacity - self.current_weight
        remaining_items = self.n - self.current_packet_idx

        return np.array([
            self.current_weight,
            remaining_capacity,
            packet["weight"],
            packet["value"],
            remaining_items
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_packet_idx = 0
        self.current_weight = 0
        self.total_value = 0
        self.selected_items = 0
        return self._get_state(), {}

    def step(self, action):
        packet = self.traffic_data[self.current_packet_idx]
        reward = 0
        done = False

        remaining_capacity = self.capacity - self.current_weight

        if action == 1:
            if packet["weight"] <= remaining_capacity:
                self.current_weight += packet["weight"]
                self.total_value += packet["value"]
                self.selected_items += 1

                # Değer / ağırlık oranını da ödüle kat
                reward = packet["value"] + 0.5 * (packet["value"] / packet["weight"])
            else:
                # Kapasiteyi aşmaya çalışma cezası
                reward = -25
        else:
            # Çok değerli item'ı boşuna geçerse hafif ceza
            ratio = packet["value"] / packet["weight"]
            reward = -2 if ratio > 8 else 0

        self.current_packet_idx += 1

        if self.current_packet_idx >= self.n:
            done = True
            # Çantayı verimli doldurduysa bonus
            usage_bonus = (self.current_weight / self.capacity) * 10
            reward += usage_bonus
            next_state = np.zeros(5, dtype=np.float32)
        else:
            next_state = self._get_state()

        return next_state, reward, done, False, {}