"""AI agents: ExpectimaxAgent (Milestone 2) and the RandomAgent baseline (Milestone 3)."""
import random
from typing import Dict, List, Optional, Tuple

from .evaluator import HeuristicEvaluator
from .game import DIRECTIONS, move_tuple

Board = List[List[int]]
LOSS_VALUE = -1_000_000.0                  # utility of a board with no legal moves
TILE_CHOICES = ((2, 0.9), (4, 0.1))        # must match Game2048 tile spawning


def _as_tuples(board) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(r) for r in board)


class ExpectimaxAgent:
    """Chooses moves by depth-limited Expectimax search.

    search_depth = number of PLAYER moves looked ahead. Each player move is
    followed by a chance node (a new 2 or 4 appears in a random empty cell).
      decision node: max over the legal moves
      chance node  : probability-weighted average over every possible tile
    At depth 0 the heuristic evaluator scores the board.

    adaptive_depth=True searches one move deeper when the board is crowded
    (4 or fewer empty cells), where there are few branches and the choice matters most.
    """

    def __init__(self, search_depth: int = 2,
                 evaluator: Optional[HeuristicEvaluator] = None,
                 adaptive_depth: bool = False):
        self.search_depth = search_depth
        self.evaluator = evaluator or HeuristicEvaluator()
        self.adaptive_depth = adaptive_depth
        self.nodes_expanded = 0
        self._decision_cache: Dict[Tuple, float] = {}
        self._chance_cache: Dict[Tuple, float] = {}

    # ------------------------------------------------------------ public API
    def choose_move(self, board: Board) -> Optional[str]:
        """Best direction for this board, or None if there is no legal move."""
        tb = _as_tuples(board)
        self._decision_cache, self._chance_cache = {}, {}     # fresh for each decision
        depth = self._effective_depth(tb)
        best_move, best_value = None, float("-inf")
        for direction in DIRECTIONS:
            new_board, _, moved = move_tuple(tb, direction)
            if not moved:
                continue
            value = self.expectimax(new_board, depth, True)
            if value > best_value:                 # strict > keeps ties deterministic
                best_move, best_value = direction, value
        return best_move

    def expectimax(self, board, depth: int, is_chance: bool) -> float:
        """Value of a board: a chance node if is_chance else a decision node."""
        tb = _as_tuples(board)
        return self._chance(tb, depth) if is_chance else self._decision(tb, depth)

    def evaluate_decision_node(self, board, depth: int) -> float:
        return self._decision(_as_tuples(board), depth)

    def evaluate_chance_node(self, board, depth: int) -> float:
        return self._chance(_as_tuples(board), depth)

    # ----------------------------------------------------------------- search
    def _effective_depth(self, tb) -> int:
        if self.adaptive_depth and sum(r.count(0) for r in tb) <= 4:
            return self.search_depth + 1
        return self.search_depth

    def _decision(self, tb, depth: int) -> float:
        """Player's turn: take the best legal move."""
        self.nodes_expanded += 1
        if depth <= 0:
            return self.evaluator.evaluate(tb)
        key = (tb, depth)
        hit = self._decision_cache.get(key)
        if hit is not None:
            return hit
        best = LOSS_VALUE                          # stays this way if no legal move
        for direction in DIRECTIONS:
            new_board, _, moved = move_tuple(tb, direction)
            if moved:
                value = self._chance(new_board, depth)
                if value > best:
                    best = value
        self._decision_cache[key] = best
        return best

    def _chance(self, tb, depth: int) -> float:
        """Random tile: average over every empty cell, 90% a 2 and 10% a 4."""
        self.nodes_expanded += 1
        key = (tb, depth)
        hit = self._chance_cache.get(key)
        if hit is not None:
            return hit
        total, cells = 0.0, 0
        for r, row in enumerate(tb):
            for c, v in enumerate(row):
                if v:
                    continue
                cells += 1
                for tile, prob in TILE_CHOICES:
                    new_row = row[:c] + (tile,) + row[c + 1:]
                    new_board = tb[:r] + (new_row,) + tb[r + 1:]   # never edits tb
                    total += prob * self._decision(new_board, depth - 1)
        result = total / cells if cells else self.evaluator.evaluate(tb)
        self._chance_cache[key] = result
        return result


class RandomAgent:
    """Baseline: picks uniformly among the legal moves."""

    def __init__(self, seed=None):
        self.random_generator = random.Random(seed)
        self.nodes_expanded = 0

    def reset(self, seed=None) -> None:
        """Re-seed so the same game seed always gives the same random player."""
        self.random_generator = random.Random(f"random-agent-{seed}")

    def choose_move(self, board: Board) -> Optional[str]:
        tb = _as_tuples(board)
        legal = [d for d in DIRECTIONS if move_tuple(tb, d)[2]]
        return self.random_generator.choice(legal) if legal else None
