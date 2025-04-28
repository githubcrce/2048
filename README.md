# 2048 Game + DQN Reinforcement Learning Agent

This project contains:
- A clean **2048 game environment** (like OpenAI Gym style)
- A **Deep Q-Network (DQN)** agent built using **PyTorch**
- A **training loop** to learn how to play 2048
- Modular, easy-to-extend code

---

## 🚀 How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Train the DQN Agent
```bash
python dqn_2048_game.py
```

Training will print average score every 100 episodes.


---

## 📦 Project Structure

```
.
├── dqn_2048_game.py     # Full Game + DQN training
├── requirements.txt      # Dependencies
├── README.md             # You're reading it!
```

---

## 🧠 How It Works

- **Environment:** 2048 board (4x4), states are normalized grids
- **Actions:** 0 = Up, 1 = Down, 2 = Left, 3 = Right
- **Rewards:** Based on increase in board tile sum; small penalty for invalid move
- **Algorithm:** Standard DQN with:
  - Experience Replay
  - Target = Bellman Equation (reward + discounted future reward)

---

## 📈 Future Upgrades

- Add Target Networks (DQN stabilization)
- Use DDQN (Double DQN)
- Prioritized Experience Replay
- Add TensorBoard logging
- Play with a trained agent (inference mode)

---

## 📜 Requirements

See `requirements.txt` below.

---

# requirements.txt
```txt
torch
numpy
```

(You only need `torch` and `numpy` for this basic version!)


---

## 🏆 Credits
Built for fun and learning reinforcement learning principles with a classic game: **2048**.

---

# ✨ Enjoy training your AI to beat 2048!
