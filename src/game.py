"""
Tic-Tac-Toe Game Environment

This module provides the game logic for tic-tac-toe, separate from
any learning algorithms. The board uses a simple representation:
- 0: Empty cell
- 1: Player X
- 2: Player O

Board positions are numbered 0-8:
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
"""

from typing import Optional


# All possible winning combinations (rows, columns, diagonals)
WIN_COMBINATIONS = [
    (0, 1, 2),  # Top row
    (3, 4, 5),  # Middle row
    (6, 7, 8),  # Bottom row
    (0, 3, 6),  # Left column
    (1, 4, 7),  # Middle column
    (2, 5, 8),  # Right column
    (0, 4, 8),  # Diagonal top-left to bottom-right
    (2, 4, 6),  # Diagonal top-right to bottom-left
]


class TicTacToe:
    """
    Tic-Tac-Toe game environment.

    Manages board state, validates moves, and detects game outcomes.
    """

    EMPTY = 0
    PLAYER_X = 1
    PLAYER_O = 2

    def __init__(self):
        """Initialize an empty board with X to move first."""
        self.reset()

    def reset(self) -> tuple:
        """
        Reset the board to initial empty state.

        Returns:
            The initial board state as a tuple.
        """
        self.board = [self.EMPTY] * 9
        self.current_player = self.PLAYER_X
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self) -> tuple:
        """
        Get the current board state.

        Returns:
            A tuple of 9 integers representing the board.
            This is hashable and can be used as a dictionary key.
        """
        return tuple(self.board)

    def get_legal_actions(self) -> list:
        """
        Get all legal moves (empty positions).

        Returns:
            List of position indices (0-8) that are empty.
        """
        return [i for i, cell in enumerate(self.board) if cell == self.EMPTY]

    def make_move(self, position: int) -> tuple:
        """
        Make a move at the specified position.

        Args:
            position: Board position (0-8) to place the current player's mark.

        Returns:
            Tuple of (new_state, reward, done, winner):
            - new_state: The board state after the move
            - reward: 1 for win, -1 for loss, 0 for draw/ongoing
            - done: True if game is over
            - winner: PLAYER_X, PLAYER_O, or None

        Raises:
            ValueError: If the position is invalid or already occupied.
        """
        if self.done:
            raise ValueError("Game is already over")
        if not 0 <= position <= 8:
            raise ValueError(f"Position must be 0-8, got {position}")
        if self.board[position] != self.EMPTY:
            raise ValueError(f"Position {position} is already occupied")

        # Make the move
        self.board[position] = self.current_player

        # Check for win
        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1  # Current player won
        # Check for draw
        elif self.EMPTY not in self.board:
            self.done = True
            self.winner = None
            reward = 0  # Draw
        else:
            reward = 0  # Game continues

        # Switch players
        previous_player = self.current_player
        self.current_player = self.PLAYER_O if self.current_player == self.PLAYER_X else self.PLAYER_X

        return self.get_state(), reward, self.done, self.winner

    def _check_win(self, player: int) -> bool:
        """Check if the specified player has won."""
        for combo in WIN_COMBINATIONS:
            if all(self.board[pos] == player for pos in combo):
                return True
        return False

    def render(self) -> str:
        """
        Create a string representation of the board.

        Returns:
            A formatted string showing the current board state.
        """
        symbols = {self.EMPTY: '.', self.PLAYER_X: 'X', self.PLAYER_O: 'O'}
        lines = []
        for row in range(3):
            cells = [symbols[self.board[row * 3 + col]] for col in range(3)]
            lines.append(' ' + ' | '.join(cells))
            if row < 2:
                lines.append('---+---+---')
        return '\n'.join(lines)

    def render_with_positions(self) -> str:
        """
        Create a string showing position numbers for empty cells.

        Returns:
            A formatted string with position hints for empty cells.
        """
        symbols = {self.PLAYER_X: 'X', self.PLAYER_O: 'O'}
        lines = []
        for row in range(3):
            cells = []
            for col in range(3):
                pos = row * 3 + col
                if self.board[pos] == self.EMPTY:
                    cells.append(str(pos))
                else:
                    cells.append(symbols[self.board[pos]])
            lines.append(' ' + ' | '.join(cells))
            if row < 2:
                lines.append('---+---+---')
        return '\n'.join(lines)

    def clone(self) -> 'TicTacToe':
        """
        Create a deep copy of the current game state.

        Returns:
            A new TicTacToe instance with the same state.
        """
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.done = self.done
        new_game.winner = self.winner
        return new_game


def play_random_game() -> Optional[int]:
    """
    Play a complete game with random moves.

    Returns:
        The winner (PLAYER_X or PLAYER_O) or None for a draw.
    """
    import random
    game = TicTacToe()
    while not game.done:
        action = random.choice(game.get_legal_actions())
        game.make_move(action)
    return game.winner


if __name__ == "__main__":
    # Demo: play a random game
    print("Playing a random game...\n")
    game = TicTacToe()
    move_num = 0

    import random
    while not game.done:
        move_num += 1
        player_name = "X" if game.current_player == TicTacToe.PLAYER_X else "O"
        action = random.choice(game.get_legal_actions())
        print(f"Move {move_num}: Player {player_name} plays position {action}")
        game.make_move(action)
        print(game.render())
        print()

    if game.winner:
        winner_name = "X" if game.winner == TicTacToe.PLAYER_X else "O"
        print(f"Player {winner_name} wins!")
    else:
        print("It's a draw!")
