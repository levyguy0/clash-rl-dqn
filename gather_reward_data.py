from env import ClashEnv
from agent import ClashAgent
import random
import torch
import time
import csv
import cv2
from pynput import keyboard

env = ClashEnv()
reward = None

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

    print("Review action.")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    # Save to CSV
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            str(state.tolist()),
            str(action),
            reward
        ])
    # i can press keys 1-0 to represent a reward of 1-10. store state, action, reward in a csv