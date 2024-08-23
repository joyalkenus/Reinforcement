import gymnasium as gym
import torch
import numpy as np
from agent import DQNAgent
import imageio
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def visualize_trained_agent(model_path, n_episodes=5, max_t=1000):
    # Set render_mode to 'rgb_array' to capture frames
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    agent = DQNAgent(state_size=8, action_size=4, seed=0)
    
    # Load the trained model
    agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=device))
    
    # Create a directory for renders if it doesn't exist
    os.makedirs('renders', exist_ok=True)

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        video_path = f'renders/trained_episode_{i_episode}.mp4'
        writer = imageio.get_writer(video_path, fps=30)

        for t in range(max_t):
            # Render and save the frame to the video
            frame = env.render()
            writer.append_data(frame)

            # Take action using the trained agent
            action = agent.act(state, eps=0.0)  # Set epsilon to 0 for greedy action
            next_state, reward, done, _, _ = env.step(action)
            state = next_state

            if done:
                break

        writer.close()
        print(f'Episode {i_episode} finished.')

    env.close()

if __name__ == "__main__":
    # Path to the saved model checkpoint
    model_path = 'models/checkpoint.pth'
    visualize_trained_agent(model_path)