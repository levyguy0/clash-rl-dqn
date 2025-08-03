import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CLASH_API_KEY")

headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

url = "https://api.clashroyale.com/v1/cards"

response = requests.get(url, headers=headers)
cards = response.json()["items"]

card_to_elixir = {item["name"].strip().lower().replace(".", "").replace("royal", "royale"): item.get("elixirCost", 0) for item in cards}
card_to_elixir["unknown"] = 0