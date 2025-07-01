import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gymnasium as gym
import torch.optim as optim
import imageio
import os
import ale_py


# --- Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, input_channels=4, num_actions=3):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, 512)
        self.policy_head = nn.Linear(512, num_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


# --- Preprocessing functions for Atari Pong (DQN style) ---
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34 : 34 + 160, :160]
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return frame


# --- Frame Stacking ---
def stack_frames(stacked_frames, new_frame, is_new_episode):
    if is_new_episode:
        stacked_frames = [new_frame for _ in range(4)]
    else:
        stacked_frames.append(new_frame)
        stacked_frames.pop(0)
    return np.stack(stacked_frames, axis=0), stacked_frames


# --- Video Saving ---
def save_video(frames, path, fps=30):
    imageio.mimsave(path, frames, fps=fps)


# --- Main Training Loop ---
def main():
    ENV_NAME = "ALE/Pong-v5"
    NUM_EPISODES = 100000
    GAMMA = 0.99
    LR = 1e-3  # Start with higher learning rate
    VIDEO_FREQ = 100
    SAVE_FREQ = 100  # or any frequency you prefer
    VIDEO_DIR = "videos_ac"
    os.makedirs(VIDEO_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    # Only use the 3 actions that matter for Pong: NOOP, UP, DOWN
    action_mapping = [0, 2, 3]
    num_actions = 3
    model = ActorCritic(input_channels=4, num_actions=num_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    CHECKPOINT_PATH = "model_ac_latest.pth"
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"Loaded model parameters from {CHECKPOINT_PATH}")

    for episode in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        frame = preprocess_frame(obs)
        stacked_frames = [frame for _ in range(4)]
        state = np.stack(stacked_frames, axis=0)
        done = False
        total_reward = 0
        log_probs = []
        values = []
        rewards = []
        entropies = []
        frames_for_video = []
        action_probs_list = []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits, value = model(state_tensor)
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # Collect action probabilities for stats
            action_probs_list.append(probs.detach().cpu().numpy()[0])

            # Map action to gym action
            gym_action = action_mapping[action.item()]
            next_obs, reward, terminated, truncated, _ = env.step(gym_action)
            done = terminated or truncated
            next_frame = preprocess_frame(next_obs)
            next_state, stacked_frames = stack_frames(stacked_frames, next_frame, False)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)
            state = next_state
            total_reward += reward

            if episode % VIDEO_FREQ == 0:
                frame_rgb = env.render()
                frames_for_video.append(frame_rgb)

        # Compute returns and losses
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(device)
        values = torch.cat(values).squeeze(1)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)

        # For actor
        advantage = returns - values.detach()

        # For critic - use MSE with returns directly
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = F.mse_loss(values, returns)  # This is cleaner
        entropy_loss = -entropies.mean()
        loss = actor_loss + critic_loss + 0.05 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        # Compute total gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # Compute action probability stats
        action_probs_arr = np.array(action_probs_list)
        mean_probs = action_probs_arr.mean(axis=0)
        std_probs = action_probs_arr.std(axis=0)
        mean_probs_str = ", ".join([f"{p:.3f}" for p in mean_probs])
        std_probs_str = ", ".join([f"{s:.3f}" for s in std_probs])

        print(
            f"Episode {episode} | Reward: {total_reward} | Loss: {loss.item():.4f} | Actor loss: {actor_loss.item():.4f} | Critic loss: {critic_loss.item():.4f} | Entropy loss: {entropy_loss.item():.4f} | Number of observations: {len(log_probs)} | Mean log prob: {log_probs.mean().item():.4f} | Mean value: {values.mean().item():.4f} | Mean entropy: {entropies.mean().item():.4f} | Grad norm: {total_norm:.4f} | Action probs mean: [{mean_probs_str}] | std: [{std_probs_str}]"
        )

        if episode % VIDEO_FREQ == 0 and frames_for_video:
            video_path = os.path.join(VIDEO_DIR, f"pong_ac_ep{episode}.mp4")
            save_video(frames_for_video, video_path)
            print(f"Saved video to {video_path}")

        if episode % SAVE_FREQ == 0:
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
