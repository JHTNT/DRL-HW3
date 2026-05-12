"""GridWorld environment used by the HW3 DQN experiments.

The behavior follows the DRL in Action Chapter 3 reference environment:

- ``static``: player, goal, pit and wall are fixed.
- ``player``: goal, pit and wall are fixed; player starts randomly.
- ``random``: all pieces start randomly.
"""

from __future__ import annotations

import numpy as np

from .gridboard import GridBoard, Position, add_tuple, rand_pair

ACTIONS = ("u", "d", "l", "r")
ACTION_TO_DELTA: dict[str, Position] = {
    "u": (-1, 0),
    "d": (1, 0),
    "l": (0, -1),
    "r": (0, 1),
}
ACTION_INDEX_TO_NAME = dict(enumerate(ACTIONS))
ACTION_NAME_TO_INDEX = {name: index for index, name in ACTION_INDEX_TO_NAME.items()}


class Gridworld:
    """Small deterministic/randomized GridWorld for DQN examples."""

    def __init__(self, size: int = 4, mode: str = "static"):
        if size < 4:
            print("Minimum board size is 4. Initialized to size 4.")
            size = 4

        self.size = size
        self.mode = mode
        self.board = GridBoard(size=size)

        # Add pieces first; positions are updated by the selected initializer.
        self.board.add_piece("Player", "P", (0, 0))
        self.board.add_piece("Goal", "+", (1, 0))
        self.board.add_piece("Pit", "-", (2, 0))
        self.board.add_piece("Wall", "W", (3, 0))

        if mode == "static":
            self.init_grid_static()
        elif mode == "player":
            self.init_grid_player()
        elif mode == "random":
            self.init_grid_rand()
        else:
            raise ValueError(f"Unknown Gridworld mode: {mode!r}")

    def init_grid_static(self) -> None:
        """Place all pieces deterministically."""

        self.board.components["Player"].pos = (0, 3)
        self.board.components["Goal"].pos = (0, 0)
        self.board.components["Pit"].pos = (0, 1)
        self.board.components["Wall"].pos = (1, 1)

    # Backward-compatible camelCase API from the reference code.
    initGridStatic = init_grid_static

    def validate_board(self) -> bool:
        """Check for overlapping or trivially blocked board layouts."""

        player = self.board.components["Player"]
        goal = self.board.components["Goal"]
        wall = self.board.components["Wall"]
        pit = self.board.components["Pit"]

        all_positions = [player.pos, goal.pos, wall.pos, pit.pos]
        if len(all_positions) > len(set(all_positions)):
            return False

        last = self.board.size - 1
        corners = [(0, 0), (0, last), (last, 0), (last, last)]
        if player.pos in corners or goal.pos in corners:
            candidate_moves = [(0, 1), (1, 0), (-1, 0), (0, -1)]
            player_moves = [self.validate_move("Player", move) for move in candidate_moves]
            goal_moves = [self.validate_move("Goal", move) for move in candidate_moves]
            if 0 not in player_moves or 0 not in goal_moves:
                return False

        return True

    validateBoard = validate_board

    def init_grid_player(self) -> None:
        """Randomize only the player location."""

        self.init_grid_static()
        self.board.components["Player"].pos = rand_pair(0, self.board.size)

        if not self.validate_board():
            self.init_grid_player()

    initGridPlayer = init_grid_player

    def init_grid_rand(self) -> None:
        """Randomize all pieces while keeping the board valid."""

        self.board.components["Player"].pos = rand_pair(0, self.board.size)
        self.board.components["Goal"].pos = rand_pair(0, self.board.size)
        self.board.components["Pit"].pos = rand_pair(0, self.board.size)
        self.board.components["Wall"].pos = rand_pair(0, self.board.size)

        if not self.validate_board():
            self.init_grid_rand()

    initGridRand = init_grid_rand

    def validate_move(self, piece: str, addpos: Position = (0, 0)) -> int:
        """Validate a move.

        Returns:
            ``0`` for a legal move, ``1`` for blocked/out of bounds, and ``2``
            for moving into the pit.
        """

        pit = self.board.components["Pit"].pos
        wall = self.board.components["Wall"].pos
        new_pos = add_tuple(self.board.components[piece].pos, addpos)

        if new_pos == wall:
            return 1
        if max(new_pos) > (self.board.size - 1):
            return 1
        if min(new_pos) < 0:
            return 1
        if new_pos == pit:
            return 2

        return 0

    validateMove = validate_move

    def make_move(self, action: str | int) -> None:
        """Move the player with action ``u``, ``d``, ``l``, ``r`` or index 0-3."""

        if isinstance(action, int):
            action = ACTION_INDEX_TO_NAME[action]

        if action not in ACTION_TO_DELTA:
            return

        addpos = ACTION_TO_DELTA[action]
        if self.validate_move("Player", addpos) in [0, 2]:
            new_pos = add_tuple(self.board.components["Player"].pos, addpos)
            self.board.move_piece("Player", new_pos)

    makeMove = make_move

    def reward(self) -> int:
        """Return the reward for the current state."""

        player_pos = self.board.components["Player"].pos
        if player_pos == self.board.components["Pit"].pos:
            return -10
        if player_pos == self.board.components["Goal"].pos:
            return 10
        return -1

    def is_terminal(self) -> bool:
        """Return whether the current state is terminal."""

        return self.reward() in {-10, 10}

    def state(self) -> np.ndarray:
        """Return the flattened one-hot state vector with shape ``(1, 64)``."""

        return self.board.render_np().reshape(1, -1).astype(np.float32)

    def display(self) -> np.ndarray:
        """Return a human-readable 2-D board."""

        return self.board.render()
