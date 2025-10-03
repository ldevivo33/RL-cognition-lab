import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------- Replay Buffer ----------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        
        states = np.stack(states).astype(np.float32)
        next_states = np.stack(next_states).astype(np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        
        dones = np.array(dones, dtype=np.float32)

        return (torch.from_numpy(states),torch.from_numpy(actions),torch.from_numpy(rewards),torch.from_numpy(next_states),torch.from_numpy(dones),)

    def __len__(self):
        return len(self.buffer)


# ---------- Neural Net ----------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------- DQN Agent ----------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.995, buffer_capacity=50000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Epsilon-greedy
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Networks
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())  # sync weights
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay memory
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.tensor([state], dtype=torch.float32)
        q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values).item())

    def update(self, batch_size=64):
        """Sample from replay buffer & update Q-network"""
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Q(s,a) for chosen actions
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Max Q for next state from target net
        next_q_values = self.target_net(next_states).max(1)[0]

        # Bellman target
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss = MSE(Q - target)
        loss = nn.MSELoss()(q_values, targets.detach())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        """Copy Q-network weights to target network"""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
