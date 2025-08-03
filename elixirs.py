import requests
import os
from dotenv import load_dotenv
import re

load_dotenv()

API_KEY = os.getenv("CLASH_API_KEY")

headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

url = "https://api.clashroyale.com/v1/cards"

response = requests.get(url, headers=headers)
cards = response.json()["items"]

def normalize_name(name):
    name = re.sub(r'\broyal\b', 'royale', name.strip().lower())
    name = name.replace(".", "")
    return name

card_to_elixir = {normalize_name(item["name"]): item.get("elixirCost", 0) for item in cards}
card_to_elixir["unknown"] = 0