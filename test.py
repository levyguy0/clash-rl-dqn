from env import ClashEnv
import torch
import random
import time
from datetime import datetime
from utils import wait_for_key

NUM_GAMES = 10

env = ClashEnv()

gamemode = input("Gamemode: ")

for episode in range(NUM_GAMES):
    while True:
        state = env.get_state()
        state_tensor = torch.tensor(state)

        if env.is_game_over():
            break

        q_values = env.agent.act(state_tensor, env.current_cards, env.available_actions)

        action_idx = torch.argmax(q_values).item()
        
        action = env.available_actions[action_idx]
        card_id, x, y = action
        card_idx = env.get_hand_idx_from_card_id(card_id)

        env.play_card(card_idx, x, y)

        time.sleep(3)
        
    # prepare for another game if there is one requested

    print(f"Complete game {episode + 1} of {NUM_GAMES}")

    if (episode + 1) != NUM_GAMES:
        if gamemode == "friendly":
            wait_for_key("0")
        else:
            time.sleep(5)
            env.start_new_game()
    else:
        print("Games finished!")
    
