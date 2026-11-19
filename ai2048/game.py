"""
2048 game engine (Milestone 1).

Two layers:
  1. Pure functions (slide_row_left, move_board) that never touch the random
     number generator. Expectimax will use these to simulate decision nodes
     and to enumerate chance nodes without changing a real game.
  2. The Game2048 class, which owns the board, score, seeded random tile
     generation and game-over detection.
"""
import random
from typing import Dict, List, Optional, Tuple

UP, DOWN, LEFT, RIGHT = "up", "down", "left", "right"
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)

Board = List[List[int]]


def slide_row_left(row: List[int]) -> Tuple[List[int], int]:
    """Slide and merge one row towards index 0.

    Each tile can merge at most once per move. Returns (new_row, score_gained).
    """
    size = len(row)
    tiles = [v for v in row if v != 0]          # 1. remove gaps
    merged: List[int] = []
    gained = 0
    i = 0
    while i < len(tiles):                        # 2. merge equal neighbours once
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            value = tiles[i] * 2
            merged.append(value)
            gained += value
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    merged += [0] * (size - len(merged))         # 3. pad with empty cells
    return merged, gained


TupleBoard = Tuple[Tuple[int, ...], ...]

# Every row only has a few possible outcomes, so each one is computed once and
# remembered. _ROW_CACHE[False] holds "slide left" results, [True] "slide right".
_ROW_CACHE: Tuple[Dict, Dict] = ({}, {})


def _slide_cached(row: Tuple[int, ...], to_right: bool):
    cache = _ROW_CACHE[to_right]
    hit = cache.get(row)
    if hit is None:
        line = list(row[::-1] if to_right else row)
        slid, gained = slide_row_left(line)
        if to_right:
            slid.reverse()
        hit = cache[row] = (tuple(slid), gained)
    return hit


def move_tuple(board: TupleBoard, direction: str) -> Tuple[TupleBoard, int, bool]:
    """Fast version of move_board for immutable (tuple of tuples) boards.

    Used by the AI search. Returns (new_board, score_gained, moved).
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"Invalid direction {direction!r}; use one of {DIRECTIONS}")
    vertical = direction in (UP, DOWN)
    to_right = direction in (RIGHT, DOWN)
    rows = tuple(zip(*board)) if vertical else board
    new_rows = []
    total = 0
    for row in rows:
        slid, gained = _slide_cached(row, to_right)
        new_rows.append(slid)
        total += gained
    new_board = tuple(zip(*new_rows)) if vertical else tuple(new_rows)
    return new_board, total, new_board != board


def move_board(board: Board, direction: str) -> Tuple[Board, int, bool]:
    """Apply a move to a board WITHOUT adding a random tile.

    Returns (new_board, score_gained, moved). The input board is not modified.
    `moved` is True only if the move changed the board (i.e. it is legal).
    """
    new_board, gained, moved = move_tuple(tuple(tuple(r) for r in board), direction)
    return [list(r) for r in new_board], gained, moved


class Game2048:
    """A 2048 game on a size x size grid (4x4 by default)."""

    def __init__(self, size: int = 4, seed: Optional[int] = None,
                 four_probability: float = 0.1):
        self.size = size
        self.four_probability = four_probability
        self.seed: Optional[int] = None
        self._rng = random.Random()
        self.board: Board = []
        self.score = 0
        self.game_over = False
        self.highest_tile = 0
        self.reset(seed)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: Optional[int] = None) -> None:
        """Start a new game. Passing a seed makes the whole game reproducible."""
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)
        self.board = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.game_over = False
        self.highest_tile = 0
        self.add_random_tile()
        self.add_random_tile()
        self._refresh_status()

    # ------------------------------------------------------------ random tile
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if self.board[r][c] == 0]

    def add_random_tile(self) -> Optional[Tuple[int, int, int]]:
        """Place a 2 (90%) or 4 (10%) in a uniformly chosen empty cell.

        Returns (row, col, value), or None if the board is full.
        """
        empty = self.get_empty_cells()
        if not empty:
            return None
        row, col = self._rng.choice(empty)
        value = 4 if self._rng.random() < self.four_probability else 2
        self.board[row][col] = value
        self.highest_tile = max(self.highest_tile, value)
        return row, col, value

    # ------------------------------------------------------------------ moves
    def move(self, direction: str) -> bool:
        """Play one move. Returns True if it was legal and was executed.

        An illegal move (one that changes nothing) leaves the game untouched
        and does not spawn a tile, exactly like the real game.
        """
        if self.game_over:
            return False
        new_board, gained, moved = move_board(self.board, direction)
        if not moved:
            return False
        self.board = new_board
        self.score += gained
        self.add_random_tile()
        self._refresh_status()
        return True

    def get_legal_moves(self) -> List[str]:
        board = tuple(tuple(r) for r in self.board)
        return [d for d in DIRECTIONS if move_tuple(board, d)[2]]

    def is_game_over(self) -> bool:
        """True when no move can change the board."""
        return len(self.get_legal_moves()) == 0

    # ---------------------------------------------------------------- helpers
    def _refresh_status(self) -> None:
        self.highest_tile = max(max(row) for row in self.board)
        self.game_over = self.is_game_over()

    def snapshot(self):
        """Save everything needed to return to this exact moment (incl. the random stream)."""
        return ([list(r) for r in self.board], self.score, self._rng.getstate())

    def restore(self, snap) -> None:
        board, score, rng_state = snap
        self.board = [list(r) for r in board]
        self.score = score
        self._rng.setstate(rng_state)
        self._refresh_status()

    def get_state(self) -> Dict:
        """Snapshot of the game (the board is a copy, safe for agents to keep)."""
        return {
            "board": [list(r) for r in self.board],
            "score": self.score,
            "game_over": self.game_over,
            "highest_tile": self.highest_tile,
        }

    def __str__(self) -> str:
        width = max(4, len(str(self.highest_tile)))
        sep = "+" + "+".join(["-" * (width + 2)] * self.size) + "+"
        lines = [sep]
        for row in self.board:
            lines.append("| " + "| ".join(
                f"{(str(v) if v else '.'):>{width}} " for v in row) + "|")
            lines.append(sep)
        lines.append(f"Score: {self.score}   Highest tile: {self.highest_tile}")
        return "\n".join(lines)
