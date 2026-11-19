"""Heuristic board evaluation.

All terms use log2 of tile values, so 2 -> 1, 4 -> 2, 8 -> 3 ... This keeps
a 1024 tile from drowning out everything else.

Smoothness and monotonicity are sums over the 8 lines (4 rows + 4 columns) of
the board. Each line's value depends only on the line itself, so it is computed
once and remembered. The weights are applied afterwards, which means that
memory stays valid while weights are being tuned.
"""
import math
from typing import Dict, List, Optional, Tuple

Board = List[List[int]]

# Hand-picked starting guess, used as the starting point (and baseline) by tune_weights.py.
INITIAL_WEIGHTS: Dict[str, float] = {
    "empty": 2.7,         # reward free cells (keeps the game alive)
    "smoothness": 0.1,    # penalise big jumps between neighbouring tiles
    "monotonicity": 1.0,  # reward rows/columns that only go up or only go down
    "corner": 1.0,        # reward the largest tile sitting in a corner
}

# Weights found by random search on development seeds (results/tuning.csv).
# On unseen dev seeds they scored +169% at depth 1 and +15% at depth 2 versus
# INITIAL_WEIGHTS (see results/tuned_weights.json).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "empty": 1.97,
    "smoothness": 0.16,
    "monotonicity": 0.27,
    "corner": 0.9,
}

_LINE_CACHE: Dict[Tuple[int, ...], Tuple[float, float]] = {}


def _log(v: int) -> float:
    return math.log2(v) if v > 0 else 0.0


def _line_stats(line: Tuple[int, ...]) -> Tuple[float, float]:
    """(smoothness, monotonicity) of one row or column. Both are <= 0, 0 is best."""
    hit = _LINE_CACHE.get(line)
    if hit is None:
        tiles = [_log(v) for v in line if v]          # empty cells are skipped
        smooth = -sum(abs(a - b) for a, b in zip(tiles, tiles[1:]))
        logs = [_log(v) for v in line]
        rising = sum(max(0.0, a - b) for a, b in zip(logs, logs[1:]))
        falling = sum(max(0.0, b - a) for a, b in zip(logs, logs[1:]))
        hit = _LINE_CACHE[line] = (smooth, -min(rising, falling))
    return hit


def _all_lines(board) -> List[Tuple[int, ...]]:
    rows = [tuple(r) for r in board]
    return rows + list(zip(*rows))


class HeuristicEvaluator:
    def __init__(self, heuristic_weights: Optional[Dict[str, float]] = None):
        self.heuristic_weights = dict(DEFAULT_WEIGHTS)
        if heuristic_weights:
            self.heuristic_weights.update(heuristic_weights)

    def evaluate(self, board) -> float:
        """Higher is better. A weighted sum of the four board features.
        Accepts lists or tuples."""
        w = self.heuristic_weights
        rows = [tuple(r) for r in board]
        smooth = mono = 0.0
        for line in rows + list(zip(*rows)):
            s, m = _line_stats(line)
            smooth += s
            mono += m
        empty = sum(r.count(0) for r in rows)
        return (w["empty"] * empty + w["smoothness"] * smooth
                + w["monotonicity"] * mono
                + w["corner"] * self.evaluate_corner_position(rows))

    def count_empty_cells(self, board) -> int:
        return sum(1 for row in board for v in row if v == 0)

    def calculate_smoothness(self, board) -> float:
        """0 is perfectly smooth; more negative means bigger jumps between
        neighbouring tiles."""
        return sum(_line_stats(line)[0] for line in _all_lines(board))

    def calculate_monotonicity(self, board) -> float:
        """0 is best. For each row and column we measure how far it is from
        being fully increasing or fully decreasing and keep the smaller gap."""
        return sum(_line_stats(line)[1] for line in _all_lines(board))

    def evaluate_corner_position(self, board) -> float:
        """+log2(max) if the largest tile is in a corner, else -log2(max)."""
        n = len(board)
        biggest = max(max(row) for row in board)
        corners = (board[0][0], board[0][n - 1], board[n - 1][0], board[n - 1][n - 1])
        return _log(biggest) if biggest in corners else -_log(biggest)
