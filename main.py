import cv2
import numpy as np
import pyautogui
from inference_sdk import InferenceHTTPClient
from agent import ClashAgent
import torch

GAME_REGION = (1175, 150, 500, 700)

class Actions:
    def __init__(self):
        pass
    

class ClashEnv:
    def __init__(self):
        self.game_region = GAME_REGION
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="jTWYejAGVnMTbcU62TBp"
        )

        self.playable_width = 500
        self.playable_height = 400
        self.num_cards = 4
        self.current_cards = []

        self.max_allies, self.max_enemies = 10, 10

        self.available_actions = self.get_available_actions()

        self.idx_to_pos = {
            0: (1300, 720),
            1: (1350, 720),
            2: (1400, 720),
            3: (1450, 720)
        }

        self.agent = ClashAgent(
            in_features=(1 + 2*10 + 2*10),
            out_features=len(self.available_actions)
        )


    def capture_game_region(self):
        screenshot = pyautogui.screenshot(region=self.game_region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    
    def get_elixir_count(self, frame):
        height, width = frame.shape[:2]

        top = int(0.90 * height)
        bottom = int(0.91 * height)
        left = int(0.28 * width)
        right = int(0.94 * width)

        elixir_bar = frame[top:bottom, left:right]

        hsv = cv2.cvtColor(elixir_bar, cv2.COLOR_BGR2HSV)

        lower_purple = np.array([125, 50, 50])
        upper_purple = np.array([160, 255, 255])
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        combined_mask = cv2.bitwise_or(purple_mask, white_mask)

        elixir_pixels = cv2.countNonZero(combined_mask)
        total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
        fill_ratio = elixir_pixels / total_pixels

        estimated_elixir = int(round(fill_ratio * 10))

        return min(estimated_elixir, 10)

    
    def get_field_info(self, frame):
        predictions = self.client.infer(frame, model_id="clash-royale-xy2jw/2")["predictions"]

        TOWER_CLASSES = [
            "ally king tower",
            "ally princess tower",
            "enemy king tower",
            "enemy princess tower"
        ]

        allies = [
            (p["x"], p["y"]) for p in predictions
            if p["class"] not in TOWER_CLASSES
            and p["class"].startswith("ally")
        ]

        enemies = [
            (p["x"], p["y"]) for p in predictions
            if p["class"] not in TOWER_CLASSES
            and p["class"].startswith("enemy")
        ]

        padded_allies = allies[:self.max_allies] + [(0, 0)] * (self.max_allies - len(allies))
        padded_enemies = enemies[:self.max_enemies] + [(0, 0)] * (self.max_enemies - len(enemies))

        norm = lambda x, y: (x / self.playable_width, y / self.playable_height)
        allies_flat = [coord for x, y in padded_allies for coord in norm(x, y)]
        enemies_flat = [coord for x, y in padded_enemies for coord in norm(x, y)]

        return allies_flat, enemies_flat
    

    def get_state(self):
        frame = self.capture_game_region()
        elixir_count = self.get_elixir_count(frame)
        allies, enemies = self.get_field_info(frame)

        state = np.array([elixir_count / 10] + allies + enemies, dtype=np.float32)

        return state
    

    def capture_cards_in_hand(self, frame):
        height, width = frame.shape[:2]

        top = int(0.80 * height)
        bottom = int(0.88 * height)
        left = int(0.28 * width)
        right = int(0.92 * width)

        card_bar = frame[top:bottom, left:right]
        
        card_width = (right - left) // 4
        cards = []

        for i in range(4):
            card_left = i * card_width
            card_right = (i + 1) * card_width
            card_img = card_bar[:, card_left:card_right]
            cv2.imshow(f"card_{i}", card_img)
            cards.append(card_img)
            

        return cards
    
    
    def get_cards_in_hand(self):
        frame = self.capture_game_region()
        cards = self.capture_cards_in_hand(frame)
    
        current_cards = []
        for card in cards:
            prediction = self.client.infer(card, model_id="cards-clash-royale-i62d3/1")
            current_cards.append(prediction["predictions"][0]["class"])

        self.current_cards = current_cards
        self.num_cards = len(current_cards)
    

    def get_available_actions(self):
        actions = [
            [card, x / (self.playable_width - 1), y / (self.playable_height - 1)]
            for card in range(self.num_cards)
            for x in range(self.playable_width)
            for y in range(self.playable_height)
        ]

        return actions
    

    def play_card(self, card_idx, x_norm, y_norm):
        card_pos = self.idx_to_pos[card_idx]
        pyautogui.moveTo(card_pos)
        pyautogui.leftClick()

        x_abs = int(self.game_region[0] + x_norm * self.playable_width)
        y_abs = int(self.game_region[1] + (1 - y_norm) * self.playable_height)

        pyautogui.dragTo(x_abs, y_abs, duration=0.2, button="left")


env = ClashEnv()