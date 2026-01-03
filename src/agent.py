"""
Q-Learning Agent for Tic-Tac-Toe

This module implements a Q-learning agent that learns to play tic-tac-toe
through experience. The agent uses:
- Q-table: A dictionary mapping (state, action) pairs to Q-values
- Epsilon-greedy: Balance between exploration and exploitation
- Temporal difference learning: Update Q-values based on rewards

Key concepts:
- Q(s, a): Expected future reward for taking action a in state s
- Learning rate (alpha): How much to update Q-values (0-1)
- Discount factor (gamma): Importance of future rewards (0-1)
- Epsilon: Probability of random exploration (0-1)
"""

import json
import random
from collections import defaultdict
from typing import Optional


class QLearningAgent:
    """
    A Q-learning agent that learns to play tic-tac-toe.

    The agent maintains a Q-table that maps (state, action) pairs to
    expected rewards. During training, it uses epsilon-greedy exploration
    and updates Q-values using the Bellman equation.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995
    ):
        """
        Initialize the Q-learning agent.

        Args:
            learning_rate: Alpha - how quickly to update Q-values (0-1)
            discount_factor: Gamma - importance of future rewards (0-1)
            epsilon: Initial exploration rate (0-1)
            epsilon_min: Minimum exploration rate
            epsilon_decay: Multiply epsilon by this after each episode
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: maps (state, action) -> Q-value
        # Using defaultdict so unknown state-actions start at 0
        self.q_table: dict = defaultdict(float)

        # For tracking learning statistics
        self.training_episodes = 0

    def get_q_value(self, state: tuple, action: int) -> float:
        """Get the Q-value for a state-action pair."""
        return self.q_table[(state, action)]

    def get_best_action(self, state: tuple, legal_actions: list) -> int:
        """
        Get the action with the highest Q-value.

        Args:
            state: Current board state
            legal_actions: List of valid moves

        Returns:
            The action with the highest Q-value (ties broken randomly)
        """
        if not legal_actions:
            raise ValueError("No legal actions available")

        # Get Q-values for all legal actions
        q_values = [(action, self.get_q_value(state, action)) for action in legal_actions]

        # Find the maximum Q-value
        max_q = max(q for _, q in q_values)

        # Get all actions with the maximum Q-value (for tie-breaking)
        best_actions = [action for action, q in q_values if q == max_q]

        return random.choice(best_actions)

    def choose_action(self, state: tuple, legal_actions: list, training: bool = True) -> int:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current board state
            legal_actions: List of valid moves
            training: If True, use epsilon-greedy; if False, always exploit

        Returns:
            The chosen action (position 0-8)
        """
        if not legal_actions:
            raise ValueError("No legal actions available")

        # Exploration: random action
        if training and random.random() < self.epsilon:
            return random.choice(legal_actions)

        # Exploitation: best known action
        return self.get_best_action(state, legal_actions)

    def learn(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        next_legal_actions: list,
        done: bool
    ):
        """
        Update Q-value using the Q-learning update rule.

        Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))

        Args:
            state: State before the action
            action: Action taken
            reward: Reward received
            next_state: State after the action
            next_legal_actions: Legal actions in next state
            done: Whether the game ended
        """
        current_q = self.get_q_value(state, action)

        if done:
            # Terminal state: no future rewards
            target = reward
        else:
            # Non-terminal: include estimated future rewards
            if next_legal_actions:
                max_next_q = max(
                    self.get_q_value(next_state, a) for a in next_legal_actions
                )
            else:
                max_next_q = 0
            target = reward + self.discount_factor * max_next_q

        # Q-learning update
        new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        """Reduce epsilon for less exploration over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def end_episode(self):
        """Called at the end of each training episode."""
        self.decay_epsilon()
        self.training_episodes += 1

    def get_stats(self) -> dict:
        """Get agent statistics for monitoring."""
        return {
            'episodes': self.training_episodes,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }

    def save(self, filepath: str):
        """
        Save the Q-table and parameters to a JSON file.

        Args:
            filepath: Path to save the agent state
        """
        # Convert Q-table keys to strings for JSON serialization
        q_table_serializable = {
            str(key): value for key, value in self.q_table.items()
        }

        data = {
            'q_table': q_table_serializable,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'training_episodes': self.training_episodes
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """
        Load the Q-table and parameters from a JSON file.

        Args:
            filepath: Path to load the agent state from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct Q-table with proper tuple keys
        self.q_table = defaultdict(float)
        for key_str, value in data['q_table'].items():
            # Parse the string key back to (tuple, int)
            # Key format: "((0, 0, 0, ...), 4)"
            key = eval(key_str)  # Safe here as we control the file format
            self.q_table[key] = value

        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        self.epsilon = data['epsilon']
        self.epsilon_min = data['epsilon_min']
        self.epsilon_decay = data['epsilon_decay']
        self.training_episodes = data['training_episodes']


class RandomAgent:
    """A simple agent that always chooses random actions."""

    def choose_action(self, state: tuple, legal_actions: list, training: bool = True) -> int:
        """Choose a random legal action."""
        return random.choice(legal_actions)


if __name__ == "__main__":
    # Demo: show how the agent works
    from game import TicTacToe

    agent = QLearningAgent()
    game = TicTacToe()

    print("Q-Learning Agent Demo")
    print("=" * 40)
    print(f"Initial epsilon: {agent.epsilon}")
    print(f"Learning rate: {agent.learning_rate}")
    print(f"Discount factor: {agent.discount_factor}")
    print()

    # Make a few moves to demonstrate
    state = game.get_state()
    legal_actions = game.get_legal_actions()

    print(f"Initial state: {state}")
    print(f"Legal actions: {legal_actions}")

    action = agent.choose_action(state, legal_actions)
    print(f"Chosen action (epsilon-greedy): {action}")
    print()

    # Simulate a learning step
    next_state, reward, done, winner = game.make_move(action)
    agent.learn(state, action, reward, next_state, game.get_legal_actions(), done)

    print(f"After learning:")
    print(f"Q({state}, {action}) = {agent.get_q_value(state, action):.4f}")
