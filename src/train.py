"""
Training Loop for Tic-Tac-Toe Q-Learning Agent

This module implements self-play training where a single agent plays
both X and O, learning from both perspectives. The agent improves by:
1. Playing against itself
2. Updating Q-values based on game outcomes
3. Gradually shifting from exploration to exploitation

Training statistics are collected for visualization.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from game import TicTacToe
from agent import QLearningAgent, RandomAgent


@dataclass
class TrainingStats:
    """Statistics collected during training."""
    episodes: list = field(default_factory=list)
    x_wins: list = field(default_factory=list)
    o_wins: list = field(default_factory=list)
    draws: list = field(default_factory=list)
    epsilons: list = field(default_factory=list)
    q_table_sizes: list = field(default_factory=list)

    # Running totals for current window
    _x_win_count: int = 0
    _o_win_count: int = 0
    _draw_count: int = 0

    def record_result(self, winner: Optional[int]):
        """Record a single game result."""
        if winner == TicTacToe.PLAYER_X:
            self._x_win_count += 1
        elif winner == TicTacToe.PLAYER_O:
            self._o_win_count += 1
        else:
            self._draw_count += 1

    def snapshot(self, episode: int, epsilon: float, q_table_size: int, window_size: int):
        """Save statistics for this checkpoint."""
        self.episodes.append(episode)
        self.x_wins.append(self._x_win_count / window_size)
        self.o_wins.append(self._o_win_count / window_size)
        self.draws.append(self._draw_count / window_size)
        self.epsilons.append(epsilon)
        self.q_table_sizes.append(q_table_size)

        # Reset running totals
        self._x_win_count = 0
        self._o_win_count = 0
        self._draw_count = 0


def train_self_play(
    agent: QLearningAgent,
    num_episodes: int = 50000,
    stats_interval: int = 1000,
    verbose: bool = True
) -> TrainingStats:
    """
    Train the agent through self-play.

    The agent plays both X and O in each game, learning from both
    perspectives. This helps the agent learn optimal play for both sides.

    Args:
        agent: The Q-learning agent to train
        num_episodes: Number of games to play
        stats_interval: How often to record statistics
        verbose: Whether to print progress

    Returns:
        TrainingStats object with training history
    """
    stats = TrainingStats()
    game = TicTacToe()

    for episode in range(1, num_episodes + 1):
        game.reset()

        # Store experience for both players to learn at game end
        # Format: (state, action, player)
        episode_history = []

        while not game.done:
            state = game.get_state()
            legal_actions = game.get_legal_actions()
            current_player = game.current_player

            # Choose action
            action = agent.choose_action(state, legal_actions, training=True)

            # Store for later learning
            episode_history.append((state, action, current_player))

            # Make move
            game.make_move(action)

        # Game is over - determine rewards and update Q-values
        winner = game.winner
        _update_from_history(agent, episode_history, winner, game)

        # Record result
        stats.record_result(winner)

        # End episode (decay epsilon)
        agent.end_episode()

        # Periodic statistics
        if episode % stats_interval == 0:
            stats.snapshot(
                episode=episode,
                epsilon=agent.epsilon,
                q_table_size=len(agent.q_table),
                window_size=stats_interval
            )

            if verbose:
                print(f"Episode {episode:>6} | "
                      f"X wins: {stats.x_wins[-1]:.1%} | "
                      f"O wins: {stats.o_wins[-1]:.1%} | "
                      f"Draws: {stats.draws[-1]:.1%} | "
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"Q-table: {len(agent.q_table)}")

    return stats


def _update_from_history(
    agent: QLearningAgent,
    history: list,
    winner: Optional[int],
    final_game: TicTacToe
):
    """
    Update Q-values for all moves in the episode.

    We work backwards through the game history, updating each move
    with the appropriate reward signal.
    """
    final_state = final_game.get_state()

    # Process moves in reverse order for proper credit assignment
    for i in range(len(history) - 1, -1, -1):
        state, action, player = history[i]

        # Determine reward from this player's perspective
        if winner is None:
            reward = 0.0  # Draw
        elif winner == player:
            reward = 1.0  # Win
        else:
            reward = -1.0  # Loss

        # Get next state info
        if i == len(history) - 1:
            # This was the last move
            next_state = final_state
            next_legal_actions = []
            done = True
        else:
            # Get state after the next move
            next_state = history[i + 1][0]
            # For next legal actions, we need to simulate - but since this is
            # self-play learning, we simplify by using the opponent's next state
            if i + 2 < len(history):
                next_legal_actions = []  # Will be filled from game state
            else:
                next_legal_actions = []
            done = False

        agent.learn(state, action, reward, next_state, next_legal_actions, done)


def evaluate_against_random(agent: QLearningAgent, num_games: int = 1000) -> dict:
    """
    Evaluate the agent against a random opponent.

    Args:
        agent: Trained Q-learning agent
        num_games: Number of evaluation games

    Returns:
        Dictionary with win/loss/draw rates as both X and O
    """
    random_agent = RandomAgent()
    results = {
        'as_x': {'wins': 0, 'losses': 0, 'draws': 0},
        'as_o': {'wins': 0, 'losses': 0, 'draws': 0}
    }

    for game_num in range(num_games):
        game = TicTacToe()

        # Alternate who plays X
        agent_is_x = (game_num % 2 == 0)

        while not game.done:
            state = game.get_state()
            legal_actions = game.get_legal_actions()

            if (game.current_player == TicTacToe.PLAYER_X) == agent_is_x:
                # Agent's turn
                action = agent.choose_action(state, legal_actions, training=False)
            else:
                # Random opponent's turn
                action = random_agent.choose_action(state, legal_actions)

            game.make_move(action)

        # Record result
        key = 'as_x' if agent_is_x else 'as_o'
        if game.winner is None:
            results[key]['draws'] += 1
        elif (game.winner == TicTacToe.PLAYER_X) == agent_is_x:
            results[key]['wins'] += 1
        else:
            results[key]['losses'] += 1

    # Convert to rates
    half_games = num_games // 2
    for key in results:
        for outcome in results[key]:
            results[key][outcome] /= half_games

    return results


def main():
    """Main training script."""
    print("Tic-Tac-Toe Q-Learning Training")
    print("=" * 50)

    # Create agent
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995
    )

    # Train
    print("\nStarting self-play training...")
    print("-" * 50)
    stats = train_self_play(agent, num_episodes=50000, stats_interval=5000)

    # Final stats
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Total episodes: {agent.training_episodes}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Q-table entries: {len(agent.q_table)}")

    # Evaluate against random
    print("\n" + "-" * 50)
    print("Evaluating against random opponent (1000 games)...")
    eval_results = evaluate_against_random(agent, num_games=1000)

    print(f"\nAs X (first player):")
    print(f"  Wins: {eval_results['as_x']['wins']:.1%}")
    print(f"  Losses: {eval_results['as_x']['losses']:.1%}")
    print(f"  Draws: {eval_results['as_x']['draws']:.1%}")

    print(f"\nAs O (second player):")
    print(f"  Wins: {eval_results['as_o']['wins']:.1%}")
    print(f"  Losses: {eval_results['as_o']['losses']:.1%}")
    print(f"  Draws: {eval_results['as_o']['draws']:.1%}")

    # Save the trained agent
    save_path = os.path.join(os.path.dirname(__file__), '..', 'trained_agent.json')
    agent.save(save_path)
    print(f"\nAgent saved to: {save_path}")

    return agent, stats


if __name__ == "__main__":
    main()
