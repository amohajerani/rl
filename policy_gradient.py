import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ale_py
import imageio
import os
import cv2

# Configuration
CONFIG = {
    # Environment settings
    "env_name": "ALE/Pong-v5",
    # Model architecture
    "input_dim": (1, 80, 80),  # Increased dimensions to better preserve the ball
    "hidden_dims": [16, 32],  # Reduced convolutional filter sizes
    "fc_dims": [128],  # Simplified fully connected layer dimensions
    "num_actions": 3,
    # Training hyperparameters
    "num_iterations": 10_000,  # Reduced number of iterations
    "num_episodes_per_batch": 2,  # Reduced episodes per batch for faster iterations
    "learning_rate": 7e-4,  # Increased learning rate for faster convergence
    "gamma": 0.99,  # Discount factor
    # Checkpointing and evaluation
    "checkpoint_freq": 500,  # Less frequent checkpointing
    "num_eval_episodes": 1,  # Reduced evaluation episodes
    "entropy_coef": 0.001,
}

# Get the script directory for consistent file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(script_dir, "pong_ai_policy.pth")
videos_dir = os.path.join(script_dir, "videos")

# Create directories if they don't exist
os.makedirs(videos_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Feature extraction for Pong state
def preprocess_frame_basic(frame):
    """
    Basic preprocessing steps for a Pong frame:
    1. Convert to grayscale
    2. Crop out the top (score) and bottom (empty space)

    Returns the preprocessed frame, which can be reused
    """
    # Convert to grayscale
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Crop out the top and bottom portions
    # The top ~35 pixels contain the score, and the bottom has empty space
    # Assuming the original Atari frame is 210x160
    frame = frame[35:190, :]  # Crop from y=35 to y=190

    return frame


def process_frame_pair(current_frame_processed, prev_frame_processed=None):
    """
    Further processing using two preprocessed frames:
    1. Calculate frame difference if prev_frame is provided
    2. Resize to target resolution
    3. Normalize pixel values

    Returns the final processed frame ready for the neural network
    """
    # Make a copy to avoid modifying the original
    frame = current_frame_processed.copy()

    # Calculate frame difference if previous frame is provided
    if prev_frame_processed is not None:
        frame = cv2.absdiff(frame, prev_frame_processed)

        # Apply slight Gaussian blur to reduce noise but preserve the ball
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        # Apply thresholding to enhance the visibility of the ball in frame differences
        _, frame = cv2.threshold(frame, 15, 255, cv2.THRESH_BINARY)

    # Resize to larger dimensions to better preserve the ball
    frame = cv2.resize(frame, (80, 80), interpolation=cv2.INTER_AREA)

    # Add channel dimension and normalize
    frame = frame.reshape(1, 80, 80) / 255.0

    return frame.astype(np.float32)


def preprocess_pong_frame(frame, prev_frame=None):
    """
    Legacy function for backward compatibility.
    Preprocess frame from Pong using the new two-step approach.
    """
    frame_processed = preprocess_frame_basic(frame)

    if prev_frame is not None:
        prev_frame_processed = preprocess_frame_basic(prev_frame)
        return process_frame_pair(frame_processed, prev_frame_processed)
    else:
        return process_frame_pair(frame_processed)


# Define a simple policy network for image inputs
class SimplePolicyNetwork(nn.Module):
    def __init__(
        self,
        input_dim=(1, 80, 80),  # Updated to match new dimensions
        hidden_dims=[16, 32],
        num_actions=3,
    ):
        super(SimplePolicyNetwork, self).__init__()

        # Simplified convolutional layers with increased stride and pooling
        self.conv1 = nn.Conv2d(input_dim[0], hidden_dims[0], kernel_size=5, stride=2)
        self.pool1 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Add pooling to reduce spatial dimensions

        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Add more pooling

        # Calculate the size of the flattened features after convolutions and pooling
        # 80x80 -> 38x38 after conv1 -> 19x19 after pool1 -> 17x17 after conv2 -> 8x8 after pool2
        conv_output_size = hidden_dims[1] * 8 * 8  # Much smaller now!

        # Smaller fully connected layer
        self.fc = nn.Linear(conv_output_size, 64)  # Reduced from 128 to 64

        # Action head (policy)
        self.action_head = nn.Linear(64, num_actions)

        # Initialize weights with simpler initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = F.relu(self.fc(x))

        # Get action probabilities
        action_probs = F.softmax(self.action_head(x), dim=1)

        return action_probs


def compute_returns(rewards, gamma=0.99):
    """Simple discounted return calculation without GAE"""
    returns = []
    G = 0
    for r in reversed(rewards):
        # For Pong, reset return estimate when point is scored
        if abs(r) == 1:
            G = r
        else:
            G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float)


# Initialize environment and policy
env = gym.make("ALE/Pong-v5")
policy = SimplePolicyNetwork().to(device)


# Calculate and print the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


total_params = count_parameters(policy)
print(f"Model has {total_params:,} trainable parameters")

optimizer = torch.optim.Adam(policy.parameters(), lr=7e-4)

# Check if the model weights are already saved, load them into the policy
if os.path.exists(checkpoint_path):
    print(f"Loading pre-trained model weights from {checkpoint_path}")
    policy.load_state_dict(torch.load(checkpoint_path))
else:
    print(f"No existing model found at {checkpoint_path}. Starting from scratch.")

# Training loop with policy gradient
for iteration in range(10_000):  # 10,000 iterations
    batch_states = []
    batch_actions = []
    batch_returns = []
    episode_rewards = []

    # Collect trajectories for a batch of episodes
    for ep in range(2):  # 2 episodes per batch
        raw_state, _ = env.reset()
        # Preprocess the initial frame
        processed_state = preprocess_frame_basic(raw_state)
        # Initialize the first frame without differencing
        state = process_frame_pair(processed_state)

        episode_states = []
        episode_actions = []
        episode_rewards = []

        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # Get action probabilities
            with torch.no_grad():
                probs = policy(state_tensor)

            # Sample action from distribution
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            # Convert to gym action (0=NOOP, 2=UP, 3=DOWN)
            gym_action = [0, 2, 3][action.item()]
            raw_next_state, reward, terminated, truncated, _ = env.step(gym_action)
            done = terminated or truncated

            # Process the next raw state (just basic preprocessing)
            processed_next_state = preprocess_frame_basic(raw_next_state)
            # Process the frame pair (current and previous)
            next_state = process_frame_pair(processed_next_state, processed_state)

            # Update previous processed state
            processed_state = processed_next_state

            # Store transition
            episode_states.append(state)
            episode_actions.append(action.item())
            episode_rewards.append(reward)

            state = next_state

        # Compute returns for the episode
        returns = compute_returns(episode_rewards, gamma=0.99)

        # Add to batch
        batch_states.extend(episode_states)
        batch_actions.extend(episode_actions)
        batch_returns.append(returns)

    # Convert lists to tensors
    batch_states = torch.FloatTensor(np.array(batch_states)).to(device)
    batch_actions = torch.LongTensor(batch_actions).to(device)
    batch_returns = torch.cat(batch_returns).to(device)

    # Normalize returns (improves training stability)
    batch_returns = (batch_returns - batch_returns.mean()) / (
        batch_returns.std() + 1e-8
    )

    # Simple policy gradient loss
    # Get current policy
    probs = policy(batch_states)
    dist = torch.distributions.Categorical(probs)
    log_probs = dist.log_prob(batch_actions)

    # Policy gradient loss: -log_prob * return
    loss = -(log_probs * batch_returns).mean()

    # Add entropy bonus to encourage exploration
    entropy = dist.entropy().mean()
    loss = loss - 0.001 * entropy  # entropy coefficient of 0.001

    # Optimize the policy
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    # Print progress
    total_rewards = sum(episode_rewards)

    # Print statement for every iteration
    print(
        f"Iteration {iteration}: Total Rewards = {total_rewards}, Loss = {loss.item():.4f}, Entropy = {entropy.item():.4f}"
    )

    # Save model checkpoint and create video every checkpoint_freq iterations
    if iteration % 100 == 0:  # Changed from CONFIG["checkpoint_freq"]
        # Save model checkpoint
        torch.save(policy.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

        # Create gameplay video
        frames = []
        eval_rewards = []
        eval_env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

        raw_state, _ = eval_env.reset()
        # Basic preprocessing for the first frame
        processed_state = preprocess_frame_basic(raw_state)
        # Initial state without differencing
        state = process_frame_pair(processed_state)

        done = False
        ep_reward = 0

        while not done:
            # Convert state to tensor and get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = policy(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            gym_action = [0, 2, 3][action.item()]

            # Take action in environment
            raw_next_state, reward, terminated, truncated, _ = eval_env.step(gym_action)
            done = terminated or truncated
            frame = eval_env.render()
            frames.append(frame)
            ep_reward += reward

            # Process the next raw state (just basic preprocessing)
            processed_next_state = preprocess_frame_basic(raw_next_state)
            # Process the frame pair
            next_state = process_frame_pair(processed_next_state, processed_state)

            # Update previous processed state
            processed_state = processed_next_state

            state = next_state

        eval_env.close()

        video_path = os.path.join(videos_dir, f"pong_ai_gameplay_iter{iteration}.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video saved at {video_path}")

# Close the environment
env.close()
