import requests
import os
import json
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

API_KEY = os.getenv("CLASH_API_KEY")

headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

url = "https://api.clashroyale.com/v1/cards"

if not (
    os.path.exists("card_to_elixir.json") and
    os.path.exists("card_to_id.json") and
    os.path.exists("card_from_id.json")
):
    print("Fetching data from API...")
    response = requests.get(url, headers=headers)
    cards = response.json()["items"]

    card_to_elixir = {
        item["name"].strip().lower().replace(".", "").replace("royal", "royale"):
        item.get("elixirCost", 0) for item in cards
    }
    card_to_elixir["unknown"] = 0

    card_to_id = {
        item["name"].strip().lower().replace(".", "").replace("royal", "royale"):
        str(item["id"] / 100000000) for item in cards
    }
    card_to_id["unknown"] = "0"

    card_from_id = {
        str(item["id"] / 100000000):
        item["name"].strip().lower().replace(".", "").replace("royal", "royale")
        for item in cards
    }
    card_from_id["0"] = "unknown"

    with open("card_to_elixir.json", "w") as f:
        json.dump(card_to_elixir, f, indent=4)

    with open("card_to_id.json", "w") as f:
        json.dump(card_to_id, f, indent=4)

    with open("card_from_id.json", "w") as f:
        json.dump(card_from_id, f, indent=4)

else:
    print("Loading data from saved files...")
    with open("card_to_elixir.json", "r") as f:
        card_to_elixir = json.load(f)

    with open("card_to_id.json", "r") as f:
        card_to_id = json.load(f)

    with open("card_from_id.json", "r") as f:
        card_from_id = json.load(f)