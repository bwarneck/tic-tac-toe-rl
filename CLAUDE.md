# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A reinforcement learning project implementing a Q-Learning agent that learns to play tic-tac-toe through self-play. The agent starts with no game knowledge and improves by updating Q-values based on game outcomes.

## Commands

```bash
# Train the agent (50,000 episodes, saves to trained_agent.json)
python src/train.py

# Play against the trained agent
python src/play.py

# Evaluate agent performance
python src/evaluate.py

# Run visualization notebook
jupyter notebook notebooks/visualization.ipynb
```

## Architecture

```
src/
├── game.py      # TicTacToe class: board state, legal moves, win detection
├── agent.py     # QLearningAgent: Q-table, epsilon-greedy, learn()
├── train.py     # Self-play training loop with statistics collection
├── evaluate.py  # Evaluation against random/minimax opponents
└── play.py      # Interactive human vs agent terminal interface
```

**Key Design Decisions:**
- Board state: tuple of 9 ints (0=empty, 1=X, 2=O), hashable for Q-table keys
- Q-table: Python dict mapping `(state, action) -> float`
- Rewards: +1 win, -1 loss, 0 draw (terminal only, no shaping)
- Exploration: Epsilon-greedy with decay (1.0 → 0.01)

## Module Interactions

1. `TicTacToe` (game.py) provides environment: `reset()`, `get_state()`, `get_legal_actions()`, `make_move()`
2. `QLearningAgent` (agent.py) learns via: `choose_action()`, `learn()`, `end_episode()`
3. `train_self_play()` (train.py) runs episodes where agent plays both X and O
4. `evaluate_agent()` (evaluate.py) tests against `RandomAgent` or `MinimaxAgent`
