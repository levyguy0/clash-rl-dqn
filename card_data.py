import requests
import os
from dotenv import load_dotenv
from pprint import pprint

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

card_to_id = {item["name"].strip().lower().replace(".", "").replace("royal", "royale"): str(item["id"] / 100000000) for item in cards}
card_to_id["unknown"] = "0"

card_from_id = {str(item["id"] / 100000000): item["name"].strip().lower().replace(".", "").replace("royal", "royale") for item in cards}
card_from_id["0"] = "unknown"