from env import ClashEnv
import random
import torch
import csv
from pynput import keyboard
from reward import RewardModel
import numpy as np

env = ClashEnv()
reward = None

reward_model = RewardModel(
    in_features=44,
    hidden_units=32,
    out_features=10
)
reward_model.load_state_dict(torch.load("reward_models/model-1754182515.604433"))

csv_path = "reward_data.csv"

def on_press(key):
    global reward
    try:
        if key.char in '123456789':
            reward = int(key.char)
            return False  # Stop listener
        elif key.char == '0':
            reward = 10
            return False
    except AttributeError:
        pass  # Ignore special keys

while not env.is_game_over():
    state = env.get_state()
    state_tensor = torch.tensor(state)

    q_values = env.agent.act(state_tensor, env.current_cards, env.available_actions)
    
    if random.random() < env.agent.epsilon:
        action_idx = random.randint(0, len(env.available_actions))
    else:
        action_idx = torch.argmax(q_values).item()
    
    action = env.available_actions[action_idx]
    card_idx, x, y = action
    env.play_card(card_idx, x, y)

    move = np.array(state.tolist() + action)
    move_tensor = torch.tensor(move, dtype=torch.float32)

    predicted_reward_logits = reward_model(move_tensor.unsqueeze(0))
    predicted_reward_probs = torch.softmax(predicted_reward_logits, dim=1)
    predicted_reward_class = torch.argmax(predicted_reward_probs, dim=1)

    print(f"Predicted reward: {predicted_reward_class + 1}")

    print("Review action.")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    # Save to CSV

def save_to_csv(state, action, reward):
    with open("reward_data.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            str(state.tolist()),
            str(action),
            reward
        ])