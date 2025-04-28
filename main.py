import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os

# --- 2048 Game Environment ---

class Game2048:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [[0 for _ in range(4)] for _ in range(4)]
        self.add_tile()
        self.add_tile()
        return self.get_state()

    def get_state(self):
        return np.array(self.board).flatten() / 2048  # normalize

    def add_tile(self):
        empty = [(i, j) for i in range(4) for j in range(4) if self.board[i][j] == 0]
        if empty:
            i, j = random.choice(empty)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def move(self, direction):
        original = [row[:] for row in self.board]

        def merge(row):
            new_row = [i for i in row if i != 0]
            for i in range(len(new_row)-1):
                if new_row[i] == new_row[i+1]:
                    new_row[i] *= 2
                    new_row[i+1] = 0
            new_row = [i for i in new_row if i != 0]
            return new_row + [0]*(4 - len(new_row))

        def move_left():
            self.board = [merge(row) for row in self.board]

        def move_right():
            self.board = [list(reversed(merge(reversed(row)))) for row in self.board]

        def transpose():
            self.board = [list(row) for row in zip(*self.board)]

        if direction == 0:  # Up
            transpose()
            move_left()
            transpose()
        elif direction == 1:  # Down
            transpose()
            move_right()
            transpose()
        elif direction == 2:  # Left
            move_left()
        elif direction == 3:  # Right
            move_right()

        reward = sum(sum(row) for row in self.board) - sum(sum(row) for row in original)

        if self.board != original:
            self.add_tile()
            done = self.is_game_over()
            return self.get_state(), reward, done
        else:
            return self.get_state(), -5, self.is_game_over()  # small penalty for invalid move

    def is_game_over(self):
        temp = [row[:] for row in self.board]
        for move in range(4):
            self.move(move)
            if temp != self.board:
                self.board = temp
                return False
        return True

# --- DQN Agent ---

class DQNAgent:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99
        self.batch_size = 128

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, 3)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        max_next_q = self.model(next_states).max(1)[0]
        expected_q = rewards + self.gamma * max_next_q * (~dones)

        loss = F.mse_loss(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# --- Training Loop ---

def train_dqn(episodes=5000):
    env = Game2048()
    agent = DQNAgent()
    scores = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.move(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)

        if ep % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {ep}, Avg Score: {avg_score:.2f}")

    return agent

# --- Start Training ---

if __name__ == "__main__":
    trained_agent = train_dqn(episodes=1000)
