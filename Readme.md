# Lunar Lander DQN

My implementation of a Deep Q-Network (DQN) to solve the Lunar Lander environment from OpenAI Gymnasium. The agent learns to control a lunar lander and safely land it on the moon's surface.


## Overview

The Lunar Lander environment is a classic reinforcement learning problem where an agent must learn to land a spacecraft safely on a lunar surface. I used openai gynasium provided lunar lander environment for this project. This project uses a Deep Q-Network (DQN) algorithm, which combines Q-learning with deep neural networks, to train an agent to perform this task.

## Implementation Details

- **Environment**: OpenAI Gymnasium's Lunar Lander-v2
- **Algorithm**: Deep Q-Network (DQN)
- **Neural Network Architecture**:
  - 3 fully connected layers
  - 64 neurons per hidden layer
  - Input: 8 (state size)
  - Output: 4 (action size)
- **Training Episodes**: 812
- **Epsilon Strategy**: Starts at 1.0, decays to 0.01
- **Replay Buffer Size**: 100,000 experiences
- **Batch Size**: 64
- **Target Network Update Frequency**: Every 4 steps
- **Optimizer**: Adam

## Installation

To run this project, you need Python 3.7 or higher. Follow these steps to set up the environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/lunar-lander-dqn.git
   cd lunar-lander-dqn
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. To train the DQN agent:
   ```bash
   python src/train.py
   ```
   This will train the agent and save the model checkpoint in the `models/` directory.

2. To visualize the trained agent's performance:
   ```bash
   python src/visualize.py
   ```
   This will generate videos of the agent's performance, saved in the `renders/` directory.

## Results

The DQN agent successfully learned to land the lunar module after 812 episodes of training. Key observations:

- **Learning Progress**: The agent showed steady improvement in its landing abilities over the course of training with some fluctuations during 400 to 500 episodes.
- **Final Performance**: By the end of training, the agent consistently achieved successful landings with high scores.
- **Stability**: The 3-layer network with 64 neurons per layer provided a good balance between learning capacity and training stability.

To see the agent in action, run the visualization script as described in the Usage section. This will generate videos demonstrating the agent's performance at different stages of training, allowing you to observe its improvement over time.