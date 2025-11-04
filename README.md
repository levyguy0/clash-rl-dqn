# Clash Royale RL Agent

An autonomous reinforcement learning agent that plays Clash Royale using deep Q-learning and computer vision.

## Overview

The Clash Royale RL Agent is a deep reinforcement learning project that learns to play Clash Royale by observing the game screen and making strategic card placement decisions. The agent uses a combination of computer vision for state perception, deep neural networks for decision-making, and reward shaping to learn optimal gameplay strategies.

## Features

- **Computer Vision Integration**: Uses Roboflow API for real-time object detection of cards, units, and towers on the battlefield
- **Deep Q-Learning**: Implements DQN (Deep Q-Network) with epsilon-greedy exploration for learning optimal card placement strategies
- **State Representation**: Captures comprehensive game state including:
  - Current elixir count
  - Positions of ally and enemy units
  - Tower health status (king and princess towers)
  - Cards currently in hand
- **Action Masking**: Prevents invalid actions such as playing cards that cost more elixir than available or placing cards in invalid positions
- **Reward Engineering**: Computes rewards based on:
  - Enemy unit eliminations
  - Tower damage dealt and received
  - Elixir efficiency
  - Game outcome (victory/defeat)
- **Model Persistence**: Automatically saves and loads trained models for continuous learning across training sessions
- **Human-in-the-Loop Training**: Optional reward model training using human-labeled data for more accurate reward prediction

## Architecture

The project consists of several key components:

- **`ClashEnv`**: Game environment that interfaces with the Clash Royale game window, captures screenshots, extracts game state, and executes actions
- **`ClashAgent`**: Reinforcement learning agent implementing DQN with action masking and reward computation
- **`RewardModel`**: Optional neural network model for predicting rewards based on state-action pairs (trainable from human feedback)
- **Training Pipeline**: Q-learning implementation with experience replay and model checkpointing

## Technologies

- **PyTorch**: Deep learning framework for neural network implementation
- **OpenCV**: Computer vision for image processing and elixir detection
- **PyAutoGUI**: Screen capture and mouse automation for interacting with the game
- **Roboflow**: Object detection API for identifying game elements
- **Clash Royale API**: Fetching card metadata and elixir costs

## Project Structure

```
clash-bot/
├── agent.py              # DQN agent implementation
├── env.py                # Game environment and state extraction
├── train.py              # Training script
├── test.py               # Testing script (greedy policy)
├── reward.py             # Reward model training
├── gather_reward_data.py  # Human reward labeling tool
├── card_data.py          # Card metadata fetching and caching
├── model_version.py      # Model checkpoint management
├── utils.py              # Utility functions
└── models/               # Saved model checkpoints
```

## Research Applications

This project demonstrates practical applications of:

- Reinforcement learning in real-time strategy games
- Computer vision for game state perception
- Reward shaping and action masking in constrained action spaces
- Human-in-the-loop machine learning for reward function design
