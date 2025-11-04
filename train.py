from env import ClashEnv
import torch
import random
import time
from datetime import datetime
from utils import wait_for_key

NUM_GAMES = 20

env = ClashEnv()

gamemode = input("Gamemode: ")

for episode in range(NUM_GAMES):
    while True:
        state = env.get_state()
        state_tensor = torch.tensor(state)

        if env.is_game_over():
            break

        q_values = env.agent.act(state_tensor, env.available_actions)

        if random.random() < env.agent.get_epsilon(episode):
            print("Playing random move")
            if random.random() < 0.1:
                action_idx = len(env.available_actions) - 1
            else:
                action_idx = random.randint(0, len(env.available_actions) - 1)
                while q_values[action_idx].item() == -1e20:
                    action_idx = random.randint(0, len(env.available_actions) - 1)
        else:
            action_idx = torch.argmax(q_values).item()
        
        action = env.available_actions[action_idx]
        card_id, x, y = action

        if card_id == -1:
            print("Doing nothing")
        else:
            card_idx = env.get_hand_idx_from_card_id(card_id)
            env.play_card(card_idx, x, y)

        time.sleep(2)

        next_state = env.get_state()
        next_state_tensor = torch.tensor(next_state)

        reward = env.agent.compute_reward(state_tensor, next_state_tensor)

        next_q_values = env.agent.act(next_state_tensor, env.available_actions)

        target = q_values.clone().detach()
        target[action_idx] = reward + env.agent.gamma * torch.max(next_q_values).item()

        loss = env.agent.loss_fn(q_values, target)
        env.agent.optimizer.zero_grad()
        loss.backward()
        env.agent.optimizer.step()

    if (episode + 1) % 5 == 0:
        torch.save(env.agent.model.state_dict(), f"models/model-{datetime.now().timestamp()}.pt")

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
    
