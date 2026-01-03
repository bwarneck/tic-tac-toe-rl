# Q-Learning Explained

A beginner-friendly guide to understanding the reinforcement learning concepts used in this project.

## The Big Picture

Imagine teaching a child to play tic-tac-toe, but instead of explaining rules, you just let them play thousands of games and tell them "good job" when they win and "try again" when they lose. Eventually, they figure out what works. That's **reinforcement learning**.

## The Key Concepts

### 1. The Agent

This is our "player" - a program that makes decisions. It starts knowing nothing about tic-tac-toe except:
- What moves are legal (empty squares)
- Whether it won, lost, or drew at the end

### 2. Q-Learning (the "brain")

The agent keeps a **cheat sheet** called a Q-table. Think of it like this:

```
"When the board looks like THIS, and I play THERE,
 how good is that move?" → a number (the Q-value)
```

For example:

| Board State | Move | Q-Value | Meaning |
|-------------|------|---------|---------|
| Empty board | Center (4) | +0.8 | "This is a good move!" |
| Empty board | Corner (0) | +0.6 | "Pretty good" |
| X in center, O threatens | Block | +0.9 | "Definitely do this" |

**Higher Q-value = better move.** The agent picks the move with the highest Q-value.

### 3. The Learning Process

After each game, the agent updates its cheat sheet:

```
"I played position 4 and eventually won →
 increase Q-value for that move"

"I played position 7 and eventually lost →
 decrease Q-value for that move"
```

The magic formula (simplified):
```
New Q = Old Q + learning_rate × (reward - Old Q)
```

### 4. Exploration vs Exploitation

Here's the dilemma: Should the agent...
- **Exploit**: Always pick the move it thinks is best? (might miss better options)
- **Explore**: Try random moves? (might learn new strategies)

Solution: **Epsilon-greedy**
- Start with lots of random moves (epsilon = 1.0 = 100% random)
- Gradually reduce randomness (epsilon decays to 0.01 = 1% random)
- Early: "Try everything!" → Later: "Use what I learned"

## How Our Code Maps to This

| File | What it does |
|------|--------------|
| `game.py` | The tic-tac-toe board - tracks state, legal moves, who won |
| `agent.py` | The Q-table + decision-making logic |
| `train.py` | Plays 50,000 games against itself, learning after each |
| `evaluate.py` | Tests: "How good did it get?" |
| `play.py` | You play against the trained agent |

## The Self-Play Trick

Our agent plays **both sides** - X and O. This way:
- It learns offense (how to win)
- It learns defense (how to block)
- It gets twice the experience per game

## Why It Works

Tic-tac-toe has about 5,478 possible board states. After 50,000 games, the agent has seen most situations multiple times and learned:
- "When I see this pattern, play here"
- "Never let opponent get two in a row unblocked"
- "Center and corners are valuable"

It discovers these strategies through trial and error, not because we programmed them.

## Key Parameters

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `learning_rate` | 0.1 | How much to update Q-values (0-1) |
| `discount_factor` | 0.95 | How much to value future rewards vs immediate |
| `epsilon` | 1.0 → 0.01 | Probability of random exploration |
| `epsilon_decay` | 0.9995 | How fast exploration decreases |
