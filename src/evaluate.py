"""
Evaluation Utilities for Tic-Tac-Toe Agent

This module provides tools to evaluate the trained agent's performance:
- Play against random opponents
- Play against optimal (minimax) opponents
- Analyze Q-table statistics
"""

from typing import Optional
from game import TicTacToe, WIN_COMBINATIONS
from agent import QLearningAgent, RandomAgent


def minimax(game: TicTacToe, maximizing: bool, alpha: float = float('-inf'), beta: float = float('inf')) -> tuple:
    """
    Minimax algorithm with alpha-beta pruning.

    Args:
        game: Current game state
        maximizing: True if maximizing player's turn
        alpha: Best value for maximizer
        beta: Best value for minimizer

    Returns:
        Tuple of (best_score, best_action)
    """
    if game.done:
        if game.winner == TicTacToe.PLAYER_X:
            return (1, None)
        elif game.winner == TicTacToe.PLAYER_O:
            return (-1, None)
        else:
            return (0, None)

    legal_actions = game.get_legal_actions()
    best_action = legal_actions[0]

    if maximizing:
        max_eval = float('-inf')
        for action in legal_actions:
            child = game.clone()
            child.make_move(action)
            eval_score, _ = minimax(child, False, alpha, beta)
            if eval_score > max_eval:
                max_eval = eval_score
                best_action = action
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return (max_eval, best_action)
    else:
        min_eval = float('inf')
        for action in legal_actions:
            child = game.clone()
            child.make_move(action)
            eval_score, _ = minimax(child, True, alpha, beta)
            if eval_score < min_eval:
                min_eval = eval_score
                best_action = action
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return (min_eval, best_action)


class MinimaxAgent:
    """An agent that plays optimally using minimax."""

    def choose_action(self, state: tuple, legal_actions: list, training: bool = True) -> int:
        """Choose the optimal action using minimax."""
        game = TicTacToe()
        game.board = list(state)

        # Determine current player from state
        x_count = sum(1 for cell in state if cell == TicTacToe.PLAYER_X)
        o_count = sum(1 for cell in state if cell == TicTacToe.PLAYER_O)
        game.current_player = TicTacToe.PLAYER_X if x_count == o_count else TicTacToe.PLAYER_O

        # For X (maximizing), we want high scores
        # For O (minimizing), we want low scores
        maximizing = (game.current_player == TicTacToe.PLAYER_X)
        _, best_action = minimax(game, maximizing)

        return best_action


def evaluate_agent(agent: QLearningAgent, opponent: str = 'random', num_games: int = 1000) -> dict:
    """
    Evaluate the agent against a specified opponent.

    Args:
        agent: The agent to evaluate
        opponent: 'random' or 'minimax'
        num_games: Number of games to play

    Returns:
        Dictionary with detailed statistics
    """
    if opponent == 'random':
        opp_agent = RandomAgent()
    elif opponent == 'minimax':
        opp_agent = MinimaxAgent()
    else:
        raise ValueError(f"Unknown opponent: {opponent}")

    results = {
        'as_x': {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0},
        'as_o': {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0}
    }

    for game_num in range(num_games):
        game = TicTacToe()
        agent_is_x = (game_num % 2 == 0)
        key = 'as_x' if agent_is_x else 'as_o'

        while not game.done:
            state = game.get_state()
            legal_actions = game.get_legal_actions()

            if (game.current_player == TicTacToe.PLAYER_X) == agent_is_x:
                action = agent.choose_action(state, legal_actions, training=False)
            else:
                action = opp_agent.choose_action(state, legal_actions)

            game.make_move(action)

        results[key]['games'] += 1
        if game.winner is None:
            results[key]['draws'] += 1
        elif (game.winner == TicTacToe.PLAYER_X) == agent_is_x:
            results[key]['wins'] += 1
        else:
            results[key]['losses'] += 1

    # Compute totals
    total = {
        'wins': results['as_x']['wins'] + results['as_o']['wins'],
        'losses': results['as_x']['losses'] + results['as_o']['losses'],
        'draws': results['as_x']['draws'] + results['as_o']['draws'],
        'games': num_games
    }

    return {
        'as_x': results['as_x'],
        'as_o': results['as_o'],
        'total': total,
        'opponent': opponent
    }


def analyze_q_table(agent: QLearningAgent) -> dict:
    """
    Analyze the Q-table for insights.

    Returns:
        Dictionary with Q-table statistics
    """
    if not agent.q_table:
        return {'error': 'Q-table is empty'}

    q_values = list(agent.q_table.values())
    states = set()
    for (state, action) in agent.q_table.keys():
        states.add(state)

    return {
        'total_entries': len(agent.q_table),
        'unique_states': len(states),
        'q_value_min': min(q_values),
        'q_value_max': max(q_values),
        'q_value_mean': sum(q_values) / len(q_values),
        'positive_q_count': sum(1 for q in q_values if q > 0),
        'negative_q_count': sum(1 for q in q_values if q < 0),
        'zero_q_count': sum(1 for q in q_values if q == 0)
    }


def print_evaluation_report(agent: QLearningAgent, verbose: bool = True):
    """Print a comprehensive evaluation report."""
    print("\n" + "=" * 60)
    print("AGENT EVALUATION REPORT")
    print("=" * 60)

    # Agent stats
    stats = agent.get_stats()
    print(f"\nTraining Statistics:")
    print(f"  Episodes trained: {stats['episodes']}")
    print(f"  Current epsilon: {stats['epsilon']:.4f}")
    print(f"  Q-table entries: {stats['q_table_size']}")

    # Q-table analysis
    q_analysis = analyze_q_table(agent)
    print(f"\nQ-Table Analysis:")
    print(f"  Unique states: {q_analysis['unique_states']}")
    print(f"  Q-value range: [{q_analysis['q_value_min']:.4f}, {q_analysis['q_value_max']:.4f}]")
    print(f"  Q-value mean: {q_analysis['q_value_mean']:.4f}")

    # Evaluate against random
    print(f"\nPerformance vs Random (1000 games):")
    random_results = evaluate_agent(agent, 'random', 1000)
    _print_results(random_results)

    # Evaluate against minimax
    print(f"\nPerformance vs Minimax (100 games):")
    minimax_results = evaluate_agent(agent, 'minimax', 100)
    _print_results(minimax_results)

    print("\n" + "=" * 60)


def _print_results(results: dict):
    """Helper to print evaluation results."""
    for role in ['as_x', 'as_o']:
        r = results[role]
        total = r['games'] if r['games'] > 0 else 1
        print(f"  {role.replace('_', ' ').title()}:")
        print(f"    Wins: {r['wins']/total:.1%} ({r['wins']}/{total})")
        print(f"    Losses: {r['losses']/total:.1%} ({r['losses']}/{total})")
        print(f"    Draws: {r['draws']/total:.1%} ({r['draws']}/{total})")

    t = results['total']
    total_games = t['games'] if t['games'] > 0 else 1
    print(f"  Overall:")
    print(f"    Wins: {t['wins']/total_games:.1%}")
    print(f"    Losses: {t['losses']/total_games:.1%}")
    print(f"    Draws: {t['draws']/total_games:.1%}")


if __name__ == "__main__":
    import os

    # Try to load a trained agent
    agent_path = os.path.join(os.path.dirname(__file__), '..', 'trained_agent.json')

    if os.path.exists(agent_path):
        print(f"Loading agent from: {agent_path}")
        agent = QLearningAgent()
        agent.load(agent_path)
        print_evaluation_report(agent)
    else:
        print(f"No trained agent found at: {agent_path}")
        print("Run train.py first to train an agent.")
