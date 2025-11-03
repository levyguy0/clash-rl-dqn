from env import ClashEnv
import torch
import random
import time
from datetime import datetime
from utils import wait_for_key

NUM_GAMES = 100

env = ClashEnv()

gamemode = input("Gamemode: ")

for episode in range(NUM_GAMES):
    while True:
        state = env.get_state()
        state_tensor = torch.tensor(state)

        if env.is_game_over():
            break

        q_values = env.agent.act(state_tensor, env.current_cards, env.available_actions)

        if random.random() < env.agent.get_epsilon(episode):
            print("Playing random move")
            action_idx = random.randint(0, len(env.available_actions))
        else:
            action_idx = torch.argmax(q_values).item()
        
        action = env.available_actions[action_idx]
        card_id, x, y = action
        card_idx = env.get_hand_idx_from_card_id(card_id)

        env.play_card(card_idx, x, y)

        time.sleep(3)

        next_state = env.get_state()
        next_state_tensor = torch.tensor(next_state)

        reward = env.agent.compute_reward(state_tensor, next_state_tensor)

        next_q_values = env.agent.act(next_state_tensor)

        target = q_values.clone().detach()
        target[action_idx] = reward + env.agent.gamma * torch.max(next_q_values).item()

        loss = env.agent.loss_fn(q_values, target)

        print(loss)

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
            env.start_new_game()
    else:
        print("Games finished!")
    
