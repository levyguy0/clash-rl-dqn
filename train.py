# to do - add to the action a card id so it learns what cards its playin.

from env import ClashEnv
import torch
import random
import time
from datetime import datetime

NUM_GAMES = 1

env = ClashEnv()

for episode in range(NUM_GAMES):
    while True:
        state = env.get_state()
        state_tensor = torch.tensor(state)

        if env.is_game_over():
            break

        q_values = env.agent.act(state_tensor, env.current_cards, env.available_actions)
        
        if random.random() < env.agent.epsilon:
            action_idx = random.randint(0, len(env.available_actions))
        else:
            action_idx = torch.argmax(q_values).item()
        
        action = env.available_actions[action_idx]
        card_id, x, y = action
        card_idx = env.get_hand_idx_from_card_id(card_id)
        env.play_card(card_idx, x, y)

        # wait a second, get the state again with env.get_state() then get reward with a manual reward function env.agent.compute_reward(state, next_state)
        # action is array of (card_id, x, y) state is [elixir, ally_x1, ally_y1, ..., enemy_x1, enemy_y1, ..., card_id1, card_id2, card_id3, card_id4, ally_king_towers, ally_princess_towers, enemy_king_towers, enemy_princess_towers]

        next_state_tensor, reward = env.agent.step(state_tensor, action, env.current_cards)
        next_q_values = env.agent.act(next_state_tensor)

        target = q_values.clone().detach()
        target[action_idx] = reward + env.agent.gamma * torch.max(next_q_values).item()

        loss = env.agent.loss_fn(q_values, target)

        print(loss)

        env.agent.optimizer.zero_grad()
        loss.backward()
        env.agent.optimizer.step()

    torch.save(env.agent.model.state_dict(), f"models/model-{datetime.now().timestamp()}.pt")

