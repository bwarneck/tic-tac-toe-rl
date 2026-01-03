"""
Human vs Agent Interactive Play

This module provides a terminal-based interface for playing tic-tac-toe
against the trained Q-learning agent.
"""

import os
import sys
from game import TicTacToe
from agent import QLearningAgent


def get_human_move(game: TicTacToe) -> int:
    """Get a valid move from the human player."""
    legal_actions = game.get_legal_actions()

    while True:
        try:
            print(f"\nYour turn! Legal moves: {legal_actions}")
            move = input("Enter position (0-8): ").strip()

            if move.lower() == 'q':
                print("Thanks for playing!")
                sys.exit(0)

            position = int(move)

            if position not in legal_actions:
                print(f"Invalid move. Choose from: {legal_actions}")
                continue

            return position

        except ValueError:
            print("Please enter a number 0-8 (or 'q' to quit)")


def show_agent_thinking(agent: QLearningAgent, state: tuple, legal_actions: list):
    """Display what the agent is thinking."""
    print("\nAgent is thinking...")

    # Show Q-values for each action
    q_values = [(action, agent.get_q_value(state, action)) for action in legal_actions]
    q_values.sort(key=lambda x: x[1], reverse=True)

    print("Q-values for each move:")
    for action, q in q_values:
        bar = "+" * int(max(0, q * 20)) if q >= 0 else "-" * int(abs(q * 20))
        print(f"  Position {action}: {q:+.3f} {bar}")

    best_action = q_values[0][0]
    print(f"\nAgent plays: {best_action}")
    return best_action


def play_game(agent: QLearningAgent, human_first: bool = True, show_thinking: bool = True):
    """
    Play a single game against the agent.

    Args:
        agent: Trained Q-learning agent
        human_first: If True, human plays X (first)
        show_thinking: If True, show agent's Q-values
    """
    game = TicTacToe()

    human_player = TicTacToe.PLAYER_X if human_first else TicTacToe.PLAYER_O
    human_symbol = "X" if human_first else "O"
    agent_symbol = "O" if human_first else "X"

    print("\n" + "=" * 40)
    print(f"NEW GAME - You are {human_symbol}")
    print("=" * 40)
    print("\nBoard positions:")
    print(" 0 | 1 | 2")
    print("---+---+---")
    print(" 3 | 4 | 5")
    print("---+---+---")
    print(" 6 | 7 | 8")
    print()

    while not game.done:
        print("\nCurrent board:")
        print(game.render())

        state = game.get_state()
        legal_actions = game.get_legal_actions()

        if game.current_player == human_player:
            # Human's turn
            action = get_human_move(game)
        else:
            # Agent's turn
            if show_thinking:
                action = show_agent_thinking(agent, state, legal_actions)
            else:
                action = agent.choose_action(state, legal_actions, training=False)
                print(f"\nAgent plays: {action}")

        game.make_move(action)

    # Game over
    print("\n" + "=" * 40)
    print("GAME OVER")
    print("=" * 40)
    print("\nFinal board:")
    print(game.render())

    if game.winner is None:
        print("\nIt's a draw!")
        return 'draw'
    elif game.winner == human_player:
        print("\nYou win! Congratulations!")
        return 'human'
    else:
        print("\nAgent wins!")
        return 'agent'


def main():
    """Main interactive play loop."""
    print("\n" + "=" * 50)
    print("    TIC-TAC-TOE: Human vs Q-Learning Agent")
    print("=" * 50)

    # Load trained agent
    agent_path = os.path.join(os.path.dirname(__file__), '..', 'trained_agent.json')

    agent = QLearningAgent()

    if os.path.exists(agent_path):
        print(f"\nLoading trained agent from: {agent_path}")
        agent.load(agent_path)
        print(f"Agent has trained for {agent.training_episodes} episodes")
        print(f"Q-table contains {len(agent.q_table)} entries")
    else:
        print(f"\nNo trained agent found at: {agent_path}")
        print("Using untrained agent (will play randomly)")

    # Game settings
    print("\n" + "-" * 50)
    human_first = input("Do you want to play first? (y/n, default=y): ").strip().lower() != 'n'
    show_thinking = input("Show agent's thinking? (y/n, default=y): ").strip().lower() != 'n'

    # Score tracking
    scores = {'human': 0, 'agent': 0, 'draw': 0}

    # Play loop
    while True:
        result = play_game(agent, human_first=human_first, show_thinking=show_thinking)
        scores[result] += 1

        print(f"\n--- Score: You {scores['human']} - Agent {scores['agent']} - Draws {scores['draw']} ---")

        again = input("\nPlay again? (y/n, default=y): ").strip().lower()
        if again == 'n':
            break

        # Alternate who goes first
        switch = input("Switch who goes first? (y/n, default=n): ").strip().lower()
        if switch == 'y':
            human_first = not human_first

    print("\n" + "=" * 50)
    print("Thanks for playing!")
    print(f"Final score: You {scores['human']} - Agent {scores['agent']} - Draws {scores['draw']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
