import gymnasium as gym
import numpy as np
import torch
from collections import deque
from agent import DQNAgent
import matplotlib.pyplot as plt
import imageio
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def train_dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    # Set render_mode to 'rgb_array' to capture frames
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    agent = DQNAgent(state_size=8, action_size=4, seed=0)
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    # List of specific episodes to save renders
    render_episodes = list(range(100, 2001, 100))

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0

        # Initialize video writer if the episode is in the render list
        if i_episode in render_episodes:
            video_path = f'renders/episode_{i_episode}.mp4'
            writer = imageio.get_writer(video_path, fps=30)

        for t in range(max_t):
            if i_episode in render_episodes:
                # Render and save the frame to the video
                frame = env.render()  
                writer.append_data(frame)

            action = agent.act(state, eps)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        # Close the video writer
        if i_episode in render_episodes:
            writer.close()

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        if np.mean(scores_window) >= 200.0:
            print(f'\nEnvironment solved in {i_episode-100} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'models/checkpoint.pth')
            break

    env.close()
    return scores

def plot_scores(scores):
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.show()

if __name__ == "__main__":
    # Create a directory for renders if it doesn't exist
    os.makedirs('renders', exist_ok=True)
    scores = train_dqn()
    plot_scores(scores)