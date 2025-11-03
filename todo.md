# CLASH BOT TODO

## 1. Field Restriction Based on Princess Towers

- When **both opponent Princess Towers are alive**,  
  set all **Q-values** corresponding to actions that would play **in the opponent’s half** of the field to **`-∞`**.  
  → This ensures the agent will **never choose those actions**, and can only act on its **own half** until a tower is destroyed.
- As soon as **one of the enemy Princess Towers is destroyed**,  
  the agent is allowed to **act anywhere on the field**.

---

## 2. Elixir-Based Action Restriction

- When choosing an action, any **card decisions** that the agent **does not have enough elixir** for should have their **Q-values** set to **`-∞`**.  
  → This ensures the agent **only plays cards it has elixir for**.

---

## 3. “Do Nothing” Action

- Add an additional action representing **doing nothing**, defined as: (-1, -1, -1)
- If the agent chooses this action:
- **No cards** should be played during this cycle.
- The action is still evaluated using the **same reward function**, since **waiting can sometimes be optimal**.

---

## 4. Clear previous saved models on run to avoid unnecessary memory hogging

- When the code is run, all files in the `models` folder are deleted, other than the most recent model which will be loaded as a checkpoint.
