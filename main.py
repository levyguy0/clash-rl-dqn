import cv2
import numpy as np
import pyautogui

GAME_REGION = (1175, 150, 500, 700)

class Actions:
    def __init__(self):
        self.game_region = GAME_REGION
        self.frame = None
    

    def capture_game_region(self):
        screenshot = pyautogui.screenshot(region=self.game_region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.frame = frame


    def get_elixir_count(self):
        height, width = self.frame.shape[:2]

        top = int(0.90 * height)
        bottom = int(0.91 * height)
        left = int(0.28 * width)
        right = int(0.94 * width)

        elixir_bar = self.frame[top:bottom, left:right]

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


actions = Actions()