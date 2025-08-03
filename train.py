from env import ClashEnv
import torch
import random
import time
from datetime import datetime

NUM_GAMES = 1

env = ClashEnv()

for episode in range(NUM_GAMES):
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

        next_state_tensor, reward = env.agent.step(state_tensor, action, env.current_cards)
        next_q_values = env.agent.act(next_state_tensor)

        target = q_values.clone().detach()
        target[action_idx] = reward + env.agent.gamma * torch.max(next_q_values).item()

        loss = env.agent.loss_fn(q_values, target)
        env.agent.optimizer.zero_grad()
        loss.backward()
        env.agent.optimizer.step()

        time.sleep(1)

    torch.save(env.agent.model.state_dict(), f"models/model-{datetime.now().timestamp()}.pt")

