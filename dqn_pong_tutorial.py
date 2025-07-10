import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
import cv2

# Simple DQN network


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 3)  # 3 actions, hard-coded

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Preprocess Pong frame: grayscale, crop, resize, normalize


def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:34+160, :160]
    frame = cv2.resize(frame, (84, 84))
    return frame.astype(np.float32) / 255.0

# Simple replay buffer


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []

    def push(self, *exp):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append(exp)

    def sample(self, batch):
        samples = random.sample(self.buffer, batch)
        return map(np.array, zip(*samples))

    def __len__(self):
        return len(self.buffer)


def stack_frames(stacked, new, new_ep):
    if new_ep:
        stacked = [new]*4
    else:
        stacked.append(new)
        stacked.pop(0)
    return np.stack(stacked, 0), stacked


def select_action(state, net, eps, device):
    if random.random() < eps:
        return random.randrange(3)  # 3 actions, hard-coded
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32,
                         device=device).unsqueeze(0)
        return net(s).argmax(1).item()


def main():
    env = gym.make('ALE/Pong-v5')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = DQN().to(device)
    tgt = DQN().to(device)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    buf = ReplayBuffer(10000)
    eps = 1.0
    gamma = 0.99
    batch = 32
    min_buf = 1000
    update_freq = 100
    steps = 0
    # Map: 0 = no-op, 1 = up, 2 = down (Pong actions: 0, 2, 5)
    action_map = [0, 2, 5]
    for ep in range(1, 201):
        obs, _ = env.reset()
        f = preprocess(obs)
        stack = [f]*4
        state = np.stack(stack, 0)
        done = False
        total = 0
        while not done:
            eps = max(0.1, eps - 1/10000)
            a = select_action(state, net, eps, device)
            env_action = action_map[a]  # Map to Pong action
            next_obs, r, term, trunc, _ = env.step(env_action)
            done = term or trunc
            nf = preprocess(next_obs)
            next_state, stack = stack_frames(stack, nf, False)
            buf.push(state, a, r, next_state, done)
            state = next_state
            total += r
            steps += 1
            if len(buf) >= min_buf:
                s, a, r, ns, d = buf.sample(batch)
                s = torch.tensor(s, dtype=torch.float32, device=device)
                a = torch.tensor(a, dtype=torch.int64, device=device)
                r = torch.tensor(r, dtype=torch.float32, device=device)
                ns = torch.tensor(ns, dtype=torch.float32, device=device)
                d = torch.tensor(d, dtype=torch.float32, device=device)
                q = net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    nq = tgt(ns).max(1)[0]
                    target = r + gamma * nq * (1 - d)
                loss = F.mse_loss(q, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if steps % update_freq == 0:
                    tgt.load_state_dict(net.state_dict())
        print(f"Episode {ep} | Reward: {total} | Epsilon: {eps:.2f}")
    env.close()


if __name__ == "__main__":
    main()
