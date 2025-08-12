import pyautogui
import numpy as np
import cv2

GAME_REGION = (1175, 150, 500, 700)

screenshot = pyautogui.screenshot(region=GAME_REGION)
frame = np.array(screenshot)
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

cv2.imshow("tower_health", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()