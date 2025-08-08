from pynput import keyboard

def wait_for_key(key_str):
    with keyboard.Events() as events:
        for event in events:
            if event.key == keyboard.KeyCode.from_char(key_str):
                break