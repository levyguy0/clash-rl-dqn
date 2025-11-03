# Clash Bot 🤖

An AI-powered reinforcement learning bot that automatically plays Clash Royale using computer vision and deep learning. The bot uses Deep Q-Networks (DQN) to learn optimal card placement strategies by observing game states through screenshots and object detection.

## 🎮 Features

- **Computer Vision Integration**: Captures game state in real-time using screenshots and object detection via Roboflow
- **Deep Q-Learning**: Uses a neural network-based agent to learn optimal gameplay strategies
- **Reward Model**: Implements a reward prediction model trained on game state transitions
- **Automatic Card Playing**: Interacts with the game using automated mouse controls
- **State Representation**: Captures elixir count, card positions, tower health, and unit positions
- **Flexible Training**: Supports both friendly battles and ladder matches

## 🏗️ Architecture

The project consists of several key components:

- **Environment (`env.py`)**: `ClashEnv` class that handles game interaction, state capture via screenshots, and object detection using Roboflow inference
- **Agent (`agent.py`)**: `ClashAgent` with a neural network model that learns Q-values for action selection
- **Reward System (`reward.py`)**: Reward model that can predict rewards for state-action pairs
- **Training Loop (`train.py`)**: Main script that runs episodes and trains the agent using Q-learning
- **Card Data (`card_data.py`)**: Fetches and manages card information from the Clash Royale API

## 📋 Prerequisites

- Python 3.13+
- Clash Royale running on your computer (emulator or desktop client)
- Roboflow account and API key
- Clash Royale API key (optional, for card data)

## 🔧 Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd clash-bot
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install pyautogui
pip install inference-sdk
pip install numpy
pip install pandas
pip install python-dotenv
pip install pynput
pip install requests
pip install torchmetrics
pip install tqdm
```

4. Set up environment variables:
   Create a `.env` file in the root directory:

```env
ROBOFLOW_API_KEY=your_roboflow_api_key_here
CLASH_API_KEY=your_clash_royale_api_key_here
```

5. Run `card_data.py` to fetch and cache card data:

```bash
python card_data.py
```

## 🚀 Usage

### Training the Bot

1. **Configure game region**: Update `GAME_REGION` in `env.py` to match your Clash Royale window position:

   ```python
   GAME_REGION = (x, y, width, height)  # Adjust to your screen
   ```

2. **Start training**:

```bash
python train.py
```

3. When prompted, enter the gamemode:
   - `friendly`: For friendly battles (waits for manual game start)
   - Any other value: For ladder/other modes (auto-starts games)

The bot will:

- Capture game state through screenshots
- Detect cards, units, and towers using object detection
- Select actions using the neural network
- Learn from rewards computed based on game state changes
- Save model checkpoints every 5 episodes

### Collecting Reward Data

To collect labeled reward data for training the reward model:

```bash
python gather_reward_data.py
```

This script will prompt you to rate actions (1-9 or 0 for 10) and save them to `reward_data.csv`.

### Training the Reward Model

```bash
python reward.py
```

This trains a reward prediction model on the collected data in `reward_data.csv`.

## 📁 Project Structure

```
clash-bot/
├── agent.py              # ClashAgent class with neural network and Q-learning logic
├── env.py                # ClashEnv class for game interaction and state capture
├── train.py              # Main training loop
├── reward.py             # Reward model training script
├── gather_reward_data.py # Script for collecting labeled reward data
├── card_data.py          # Fetches and caches card data from Clash Royale API
├── utils.py              # Utility functions
├── model_version.py      # Helper to find most recent model checkpoint
├── test.py               # Simple test script for screenshots
├── models/               # Saved model checkpoints
├── reward_models/        # Saved reward model checkpoints
├── reward_data.csv       # Collected reward training data
├── card_to_elixir.json   # Card name to elixir cost mapping
├── card_to_id.json       # Card name to ID mapping
├── card_from_id.json     # Card ID to name mapping
├── pyproject.toml        # Project metadata
└── README.md             # This file
```

## 🧠 How It Works

### State Representation

The bot's state vector includes:

- Elixir count (normalized 0-1)
- Ally unit positions (normalized coordinates, max 10 units)
- Enemy unit positions (normalized coordinates, max 10 units)
- Tower counts (ally/enemy king/princess towers)
- Current hand cards (encoded as IDs)

Total state size: `1 + 2*10 + 2*10 + 4 + 2 + 2 = 45 features`

### Action Space

The action space consists of all possible card placements:

- 4 cards in hand
- 500 × 400 possible positions on the field
- Total: ~800,000 possible actions per state

### Reward Function

The reward is computed based on:

- Enemy units eliminated: +5 per unit
- Ally units lost: -3 per unit
- Enemy princess towers destroyed: +20 per tower
- Ally princess towers lost: -20 per tower
- Enemy king tower destroyed: +50
- Ally king tower destroyed: -50
- Elixir efficiency: bonus for efficient trades

### Training Process

1. Agent observes current game state
2. Selects action using ε-greedy policy (exploration vs exploitation)
3. Executes action (plays card at position)
4. Observes new state and computes reward
5. Updates Q-network using Q-learning loss
6. Repeats until game ends

## ⚙️ Configuration

Key hyperparameters in `agent.py`:

- `epsilon_start`: 0.6 (initial exploration rate)
- `epsilon_min`: 0.01 (minimum exploration rate)
- `decay_rate`: 0.001 (epsilon decay)
- `gamma`: 0.95 (discount factor)
- Learning rate: 0.01
- Hidden units: 10 (for main model), 32 (for reward model)

## 🐛 Known Limitations

- Game region coordinates are hardcoded and need manual adjustment
- Action space is very large (800k actions), making training challenging
- No action masking for invalid moves (see `todo.md` for planned improvements)
- Requires manual game setup and window positioning

## 📝 TODO

See `todo.md` for planned improvements:

- Field restriction based on princess towers
- Elixir-based action restriction
- "Do Nothing" action option

## ⚠️ Disclaimer

This project is for educational and research purposes only. Using automated bots to play Clash Royale may violate the game's terms of service. Use at your own risk.
