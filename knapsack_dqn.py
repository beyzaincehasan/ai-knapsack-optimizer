import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQNNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=10000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQNNet(state_size, action_size).to(self.device)
        self.target_model = DQNNet(state_size, action_size).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            act_values = self.model(state)

        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            state_t = torch.FloatTensor(state).to(self.device)
            next_state_t = torch.FloatTensor(next_state).to(self.device)

            target = reward
            if not done:
                with torch.no_grad():
                    future_q = torch.max(self.target_model(next_state_t.unsqueeze(0))).item()
                    target = reward + self.gamma * future_q

            current_q = self.model(state_t.unsqueeze(0)).detach().clone()
            current_q[0][action] = target

            states.append(state_t)
            targets.append(current_q.squeeze(0))

        states = torch.stack(states)
        targets = torch.stack(targets)

        outputs = self.model(states)
        loss = F.mse_loss(outputs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay