import cv2
import numpy as np
import pyautogui
from inference_sdk import InferenceHTTPClient

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

        allies_flat = [coord for pos in allies for coord in pos]
        enemies_flat = [coord for pos in enemies for coord in pos]

        return allies_flat, enemies_flat
    

    def get_state(self):
        frame = self.capture_game_region()
        elixir_count = self.get_elixir_count(frame)
        allies, enemies = self.get_field_info(frame)

        state = np.array([elixir_count / 10] + allies + enemies, dtype=np.float32)

        return state
    

    def get_available_cards(self):
        pass
    


env = ClashEnv()
env.get_state()