# tic-tac-toe-rl

A reinforcement learning project exploring how an AI agent learns to play tic-tac-toe optimally through self-play and environmental interaction.

## Overview

This project implements a Q-Learning agent that learns to play tic-tac-toe by repeatedly playing games and updating its strategy based on outcomes. The agent starts with no knowledge of the game and improves over time to maximize winning and minimize losing.

This is a public learning workshop—documenting the journey of building RL agents from first principles, including experiments, iterations, and lessons learned along the way.

## Motivation

Tic-tac-toe serves as an ideal starting point for reinforcement learning because it:
- Has a discrete, finite action space (9 possible moves)
- Has clear, unambiguous outcomes (win, loss, draw)
- Is simple enough to understand fully, yet complex enough to demonstrate RL concepts
- Allows for exhaustive exploration and verification of learned strategies
- Provides immediate feedback for learning

## Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/bwarneck/tic-tac-toe-rl.git
cd tic-tac-toe-rl
pip install jupyter matplotlib numpy
```

### Run the Training Notebook

The easiest way to train the agent and visualize results is through the Jupyter notebook:

```bash
jupyter notebook notebooks/visualization.ipynb
```

Run each cell in order to:
1. Train a Q-learning agent through self-play
2. Visualize win/loss/draw rates over training episodes
3. Analyze Q-table statistics and agent performance

### Command Line Scripts

Alternatively, use the command line scripts:

```bash
# Train the agent (50,000 episodes, saves to trained_agent.json)
python src/train.py

# Play against the trained agent
python src/play.py

# Evaluate agent performance against random/minimax opponents
python src/evaluate.py
```

## Contributing

This is a learning project, but feedback and discussion are welcome! Feel free to open issues with questions, suggestions, or observations.

## License

MIT

## Author

Built as part of a public learning workshop in AI and reinforcement learning.
