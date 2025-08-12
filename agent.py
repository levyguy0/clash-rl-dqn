import torch
from torch import nn
from card_data import card_to_elixir
from pynput import keyboard
from reward import RewardModel
import numpy as np
from model_version import get_recent_model
import csv
import math


class ClashModel(nn.Module):
    def __init__(self, in_features, hidden_units, out_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=out_features)
        )

    
    def forward(self, x):
        x = self.layers(x)
        return x


class ClashAgent:
    def __init__(self, in_features, out_features):
        self.model = ClashModel(in_features=in_features,
                   hidden_units=10,
                   out_features=out_features)
        self.epsilon_start = 0.6
        self.epsilon_min = 0.01
        self.decay_rate = 0.001
        self.gamma = 0.95
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        self.reward = None
        self.reward_model = RewardModel(
            in_features=44,
            hidden_units=32,
            out_features=10
        )

        reward_model_path = get_recent_model("reward_models")
        if reward_model_path != "None":
            self.reward_model.load_state_dict(torch.load(reward_model_path))

    
    def get_epsilon(self, episode):
        return self.epsilon_min + (self.epsilon_start - self.epsilon_min) * math.exp(-self.decay_rate * episode)


    def act(self, state, current_cards=None, available_actions=None):
        q_values = self.model(state)

        # if current_cards is not None and available_actions is not None:
        #     # set any where the corresponding action is a card for too much elixir or placed in invalid position to -inf
        #     mask = torch.ones(len(available_actions), dtype=torch.bool)

        #     for i, (card_idx, x, y) in enumerate(available_actions):
        #         card_name = current_cards[card_idx]
        #         elixir_cost = card_to_elixir[card_name]

        #         elixir_available = state[0].item() * 10

        #         if elixir_cost > elixir_available:
        #             mask[i] = False

        #     q_values[~mask] = float('-inf')

        return q_values
    

    # def step(self, state, action, current_cards):
    #     card_idx, x, y = action
    #     card_name = current_cards[card_idx]

    #     elixir_cost = card_to_elixir[card_name]

    #     next_state = state.clone()
    #     next_state[0] = max(0.0, next_state[0] - (elixir_cost / 10))

    #     ally_section = next_state[1:21].reshape(10, 2)
    #     for i in range(len(ally_section)):
    #         curr_x, curr_y = ally_section[i]

    #         if curr_x == 0 and curr_y == 0:
    #             ally_section[i] = torch.tensor([x, y])
    #             break
    #     else:
    #         ally_section[-1] = torch.tensor([x, y])

    #     ally_section = ally_section.reshape(1, 20)

    #     next_state[1:21] = ally_section

    #     reward = self.compute_reward(state, action)

    #     return next_state, reward
    

    # def await_manual_reward(self, key):
    #     try:
    #         if key.char in '123456789':
    #             self.reward = int(key.char)
    #             return False  # Stop listener
    #         elif key.char == '0':
    #             self.reward = 10
    #             return False
    #     except AttributeError:
    #         pass  # Ignore special keys

    
    # def save_to_csv(self, state, action, reward):
    #     with open("reward_data.csv", mode='a', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow([
    #             str(state.tolist()),
    #             str(action),
    #             reward
    #         ])
    

    def compute_reward(self, prev_state, new_state):
        # manual reward training

        # print("Review action.")
        # with keyboard.Listener(on_press=self.await_manual_reward) as listener:
        #     listener.join()
        
        # self.save_to_csv(state, action, self.reward)

        # print(f"Given reward: {self.reward}")

        # reward model predicted reward

        # move = np.array(state.tolist() + action)
        # move_tensor = torch.tensor(move, dtype=torch.float32)
        # predicted_reward_logits = self.reward_model(move_tensor.unsqueeze(0))
        # predicted_reward_probs = torch.softmax(predicted_reward_logits, dim=1)
        # predicted_reward_class = torch.argmax(predicted_reward_probs, dim=1)

        # print(f"Predicted reward: {predicted_reward_class.item() + 1}")

        # return self.reward

        # calculate reward here
        prev_data = {
            "elixir": (prev_state[0] * 10).item(),
            "ally_king_tower": prev_state[41].item(),
            "enemy_king_tower": prev_state[42].item(),
            "ally_princess_towers": prev_state[43].item(),
            "enemy_princess_towers": prev_state[44].item()
        }

        prev_ally_count = 0
        prev_enemy_count = 0

        prev_ally_section = prev_state[1:21]
        for coord in prev_ally_section:
            if coord != 0:
                prev_ally_count += 1

        prev_ally_count = math.ceil(prev_ally_count / 2)

        prev_enemy_section = prev_state[21:41]
        for coord in prev_enemy_section:
            if coord != 0:
                prev_enemy_count += 1

        prev_enemy_count = math.ceil(prev_enemy_count / 2)

        prev_data["ally_count"] = prev_ally_count
        prev_data["enemy_count"] = prev_enemy_count

        new_data = {
            "elixir": (new_state[0] * 10).item(),
            "ally_king_tower": new_state[41].item(),
            "enemy_king_tower": new_state[42].item(),
            "ally_princess_towers": new_state[43].item(),
            "enemy_princess_towers": new_state[44].item()
        }

        new_ally_count = 0
        new_enemy_count = 0

        new_ally_section = new_state[1:21]
        for coord in new_ally_section:
            if coord != 0:
                new_ally_count += 1

        new_ally_count = math.ceil(new_ally_count / 2)

        new_enemy_section = new_state[21:41]
        for coord in new_enemy_section:
            if coord != 0:
                new_enemy_count += 1

        new_enemy_count = math.ceil(new_enemy_count / 2)

        new_data["ally_count"] = new_ally_count
        new_data["enemy_count"] = new_enemy_count

        # calculate reward

        enemy_reward = 0
        ally_reward = 0
        elixir_reward = 0
        enemy_princess_reward = 0
        ally_princess_reward = 0
        ally_king_reward = 0 
        enemy_king_reward = 0
        
        if new_data["enemy_count"] < prev_data["enemy_count"]:
            diff = prev_data["enemy_count"] - new_data["enemy_count"]
            enemy_reward += diff * 5

            elixir_spent = prev_data["elixir"] - new_data["elixir"]
            if elixir_spent > 0:
                elixir_reward += (diff - elixir_spent)

        if new_data["ally_count"] < prev_data["ally_count"]:
            diff = prev_data["ally_count"] - new_data["ally_count"]

            ally_reward -= diff * 3

        if new_data["enemy_princess_towers"] < prev_data["enemy_princess_towers"]:
            diff = prev_data["enemy_princess_towers"] - new_data["enemy_princess_towers"]
            enemy_princess_reward += diff * 20

        if new_data["ally_princess_towers"] < prev_data["ally_princess_towers"]:
            diff = prev_data["ally_princess_towers"] - new_data["ally_princess_towers"]

            ally_princess_reward -= diff * 20

        if new_data["ally_king_tower"] == 0:

            ally_king_reward -= 50

        if new_data["enemy_king_tower"] == 0:

            enemy_king_reward += 50

        reward = enemy_reward + ally_reward + elixir_reward + enemy_princess_reward + ally_princess_reward + enemy_king_reward + ally_king_reward

        print(f"Enemy reward: {enemy_reward}\nAlly reward: {ally_reward}\nEnemy princess reward: {enemy_princess_reward}\nAlly princess reward: {ally_princess_reward}\nEnemy king reward: {enemy_king_reward}\nAlly king reward: {ally_king_reward}\nElixir reward: {elixir_reward}\nTotal reward: {reward}")
        
        return reward

        

