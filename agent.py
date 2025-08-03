import torch
from torch import nn
from elixirs import card_to_elixir
from pynput import keyboard
from reward import RewardModel
import numpy as np
from model_version import get_recent_model
import csv


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
        self.epsilon = 0.1
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
    

    def step(self, state, action, current_cards):
        card_idx, x, y = action
        card_name = current_cards[card_idx]

        elixir_cost = card_to_elixir[card_name]

        next_state = state.clone()
        next_state[0] = max(0.0, next_state[0] - (elixir_cost / 10))

        ally_section = next_state[1:21].reshape(10, 2)
        for i in range(len(ally_section)):
            curr_x, curr_y = ally_section[i]

            if curr_x == 0 and curr_y == 0:
                ally_section[i] = torch.tensor([x, y])
                break
        else:
            ally_section[-1] = torch.tensor([x, y])

        ally_section = ally_section.reshape(1, 20)

        next_state[1:21] = ally_section

        reward = self.compute_reward(state, action)

        return next_state, reward
    

    def await_manual_reward(self, key):
        try:
            if key.char in '123456789':
                self.reward = int(key.char)
                return False  # Stop listener
            elif key.char == '0':
                self.reward = 10
                return False
        except AttributeError:
            pass  # Ignore special keys

    
    def save_to_csv(self, state, action, reward):
        with open("reward_data.csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                str(state.tolist()),
                str(action),
                reward
            ])
    

    def compute_reward(self, state, action):
        # manual reward training

        print("Review action.")
        with keyboard.Listener(on_press=self.await_manual_reward) as listener:
            listener.join()
        
        self.save_to_csv(state, action, self.reward)

        print(f"Given reward: {self.reward}")

        # reward model predicted reward

        move = np.array(state.tolist() + action)
        move_tensor = torch.tensor(move, dtype=torch.float32)
        predicted_reward_logits = self.reward_model(move_tensor.unsqueeze(0))
        predicted_reward_probs = torch.softmax(predicted_reward_logits, dim=1)
        predicted_reward_class = torch.argmax(predicted_reward_probs, dim=1)

        print(f"Predicted reward: {predicted_reward_class.item() + 1}")

        return self.reward
        

