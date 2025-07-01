import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random
import gymnasium as gym
import torch.optim as optim
import imageio
import os
import ale_py


class DQN(nn.Module):
    def __init__(self, input_channels=4, num_actions=6):
        super(DQN, self).__init__()
        # Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Example usage:
# env = gym.make('ALE/Pong-v5')
# dqn = DQN(input_channels=4, num_actions=env.action_space.n)


# --- Preprocessing functions for Atari Pong (DQN style) ---
def preprocess_frame(frame):
    # Convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Crop and resize to 84x84
    frame = frame[34 : 34 + 160, :160]  # Crop to 160x160
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    # Normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    return frame  # shape: (84, 84)


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.state_shape = state_shape
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size


# --- DQN Training and Evaluation for Pong ---


def stack_frames(stacked_frames, new_frame, is_new_episode):
    if is_new_episode:
        stacked_frames = [new_frame for _ in range(4)]
    else:
        stacked_frames.append(new_frame)
        stacked_frames.pop(0)
    return np.stack(stacked_frames, axis=0), stacked_frames


def select_action(state, policy_net, epsilon, num_actions, device):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state)
            return q_values.argmax(1).item()


def save_video(frames, path, fps=30):
    imageio.mimsave(path, frames, fps=fps)


def main():
    # --- Hyperparameters ---
    ENV_NAME = "ALE/Pong-v5"
    NUM_EPISODES = 1000
    REPLAY_SIZE = 100_000
    BATCH_SIZE = 32
    GAMMA = 0.99
    LEARNING_RATE = 1e-4
    TARGET_UPDATE_FREQ = 10
    EPS_START = 1.0
    EPS_END = 0.1
    EPS_DECAY = 500_000  # steps
    MIN_REPLAY_SIZE = 10_000
    VIDEO_FREQ = 100  # Save video every N episodes
    SAVE_PATH = "pong_dqn.pth"
    VIDEO_DIR = "videos"
    os.makedirs(VIDEO_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    num_actions = env.action_space.n

    policy_net = DQN(input_channels=4, num_actions=num_actions).to(device)
    target_net = DQN(input_channels=4, num_actions=num_actions).to(device)

    # Load model parameters if SAVE_PATH exists
    if os.path.exists(SAVE_PATH):
        policy_net.load_state_dict(torch.load(SAVE_PATH, map_location=device))
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Loaded model parameters from {SAVE_PATH}")
    else:
        target_net.load_state_dict(policy_net.state_dict())
        print("Initialized model from scratch.")

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_SIZE, (4, 84, 84))

    steps_done = 0
    episode_rewards = []
    epsilon = EPS_START

    for episode in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        frame = preprocess_frame(obs)
        stacked_frames = [frame for _ in range(4)]
        state = np.stack(stacked_frames, axis=0)
        done = False
        total_reward = 0
        frames_for_video = []
        episode_losses = []  # Track losses for this episode

        while not done:
            epsilon = max(EPS_END, EPS_START - steps_done / EPS_DECAY)
            action = select_action(state, policy_net, epsilon, num_actions, device)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_frame = preprocess_frame(next_obs)
            next_state, stacked_frames = stack_frames(stacked_frames, next_frame, False)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps_done += 1

            # For video
            if episode % VIDEO_FREQ == 0:
                frame_rgb = env.render()
                frames_for_video.append(frame_rgb)

            # Training
            if len(replay_buffer) >= MIN_REPLAY_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(
                    BATCH_SIZE
                )
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    target = rewards + GAMMA * next_q_values * (1 - dones)
                loss = torch.nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episode_losses.append(loss.item())  # Track loss

                # Update target network
                if steps_done % TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(total_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else None
        if avg_loss is not None:
            print(
                f"Episode {episode} | Reward: {total_reward} | Epsilon: {epsilon:.3f} | Loss: {avg_loss:.4f}"
            )
        else:
            print(
                f"Episode {episode} | Reward: {total_reward} | Epsilon: {epsilon:.3f} | Loss: N/A"
            )

        # Save model
        if episode % VIDEO_FREQ == 0:
            torch.save(policy_net.state_dict(), SAVE_PATH)
            print(f"Saved model to {SAVE_PATH}")
            if frames_for_video:
                video_path = os.path.join(VIDEO_DIR, f"pong_dqn_ep{episode}.mp4")
                save_video(frames_for_video, video_path)
                print(f"Saved video to {video_path}")

    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
