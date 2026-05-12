"""Board primitives for the DRL in Action GridWorld environment.

This module is adapted from the Chapter 3 reference implementation and kept
small so the DQN scripts can import it as a normal package module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Position = tuple[int, int]


def rand_pair(start: int, end: int) -> Position:
    """Return a random ``(row, col)`` pair in ``[start, end)``."""

    return int(np.random.randint(start, end)), int(np.random.randint(start, end))


# Backward-compatible name used by the original book code.
randPair = rand_pair


@dataclass
class BoardPiece:
    """A movable or stationary item on the board."""

    name: str
    code: str
    pos: Position


@dataclass
class BoardMask:
    """A fixed mask layer on the board, such as a boundary."""

    name: str
    mask: np.ndarray
    code: str

    def get_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return row and column arrays for non-zero mask positions."""

        return np.nonzero(self.mask)


def zip_positions2d(positions: tuple[np.ndarray, np.ndarray]) -> list[Position]:
    """Convert numpy ``nonzero`` output to a list of ``(row, col)`` tuples."""

    rows, cols = positions
    return list(zip(rows.tolist(), cols.tolist()))


class GridBoard:
    """A square board containing named pieces and optional mask layers."""

    def __init__(self, size: int = 4):
        self.size = size
        self.components: dict[str, BoardPiece] = {}
        self.masks: dict[str, BoardMask] = {}

    def add_piece(self, name: str, code: str, pos: Position = (0, 0)) -> None:
        self.components[name] = BoardPiece(name, code, pos)

    # Backward-compatible camelCase API from the reference code.
    addPiece = add_piece

    def add_mask(self, name: str, mask: np.ndarray, code: str) -> None:
        self.masks[name] = BoardMask(name, mask, code)

    addMask = add_mask

    def move_piece(self, name: str, pos: Position) -> None:
        move = True
        for mask in self.masks.values():
            if pos in zip_positions2d(mask.get_positions()):
                move = False
                break
        if move:
            self.components[name].pos = pos

    movePiece = move_piece

    def del_piece(self, name: str) -> None:
        del self.components[name]

    delPiece = del_piece

    def render(self) -> np.ndarray:
        """Return a 2-D human-readable board with piece codes."""

        dtype = "<U2"
        display_board = np.zeros((self.size, self.size), dtype=dtype)
        display_board[:] = " "

        for piece in self.components.values():
            display_board[piece.pos] = piece.code

        for mask in self.masks.values():
            display_board[mask.get_positions()] = mask.code

        return display_board

    def render_np(self) -> np.ndarray:
        """Return one-hot layers with shape ``(num_layers, size, size)``."""

        num_pieces = len(self.components) + len(self.masks)
        display_board = np.zeros((num_pieces, self.size, self.size), dtype=np.uint8)

        layer = 0
        for piece in self.components.values():
            pos = (layer,) + piece.pos
            display_board[pos] = 1
            layer += 1

        for mask in self.masks.values():
            rows, cols = mask.get_positions()
            layers = np.repeat(layer, len(rows))
            display_board[(layers, rows, cols)] = 1
            layer += 1

        return display_board


def add_tuple(a: Position, b: Position) -> Position:
    """Element-wise addition for board positions."""

    return tuple(sum(values) for values in zip(a, b))  # type: ignore[return-value]


# Backward-compatible name used by the original book code.
addTuple = add_tuple
