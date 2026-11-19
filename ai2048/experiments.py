"""Fixed-seed experiments (Milestones 3 and 4): ExperimentRunner and helpers."""
import csv
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional

from .game import Game2048

FIELDS = ["agent", "seed", "final_score", "highest_tile", "moves", "reached_2048",
          "decision_time_s", "avg_decision_ms", "nodes_expanded"]


def make_seeds(n: int = 100, master_seed: int = 2048, low: int = 1, high: int = 1_000_000) -> List[int]:
    """n distinct game seeds drawn from a fixed master seed (same list every time)."""
    return random.Random(master_seed).sample(range(low, high), n)


def play_game(agent, seed: int) -> Dict:
    """Play one full game with the given agent and return its statistics."""
    game = Game2048(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset(seed)
    agent.nodes_expanded = 0
    moves, decision = 0, 0.0
    while not game.game_over:
        t0 = time.perf_counter()
        move = agent.choose_move(game.board)
        decision += time.perf_counter() - t0
        if move is None or not game.move(move):      # safety: never loop forever
            break
        moves += 1
    return {"seed": seed, "final_score": game.score, "highest_tile": game.highest_tile,
            "moves": moves, "reached_2048": game.highest_tile >= 2048,
            "decision_time_s": decision, "avg_decision_ms": 1000 * decision / max(moves, 1),
            "nodes_expanded": agent.nodes_expanded}


def _run_task(task):
    name, factory, seed = task
    record = play_game(factory(), seed)
    record["agent"] = name
    return record


def load_results(path: str) -> List[Dict]:
    """Read a games CSV written by export_results / run_experiments."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"agent": r["agent"], "seed": int(r["seed"]),
                         "final_score": int(r["final_score"]),
                         "highest_tile": int(r["highest_tile"]), "moves": int(r["moves"]),
                         "reached_2048": r["reached_2048"] == "True",
                         "decision_time_s": float(r["decision_time_s"]),
                         "avg_decision_ms": float(r["avg_decision_ms"]),
                         "nodes_expanded": int(r["nodes_expanded"])})
    return rows


class ExperimentRunner:
    """Runs every agent on every test seed and summarises the results.

    agents     : {name: factory}; a factory is a class or functools.partial that
                 builds a fresh agent, e.g. partial(ExpectimaxAgent, 2).
    test_seeds : the game seeds; every agent plays exactly the same seeds.
    results    : one record (dict) per finished game.
    """

    def __init__(self, agents: Dict[str, Callable], test_seeds: Iterable[int],
                 results: Optional[List[Dict]] = None):
        self.agents = dict(agents)
        self.test_seeds = list(test_seeds)
        self.results: List[Dict] = list(results or [])

    def run_game(self, name: str, seed: int) -> Dict:
        record = play_game(self.agents[name](), seed)
        record["agent"] = name
        return record

    def run_experiments(self, workers: int = 1, csv_path: Optional[str] = None,
                        progress: Optional[Callable] = None) -> List[Dict]:
        """Play all (agent, seed) pairs that are not finished yet.

        If csv_path exists its games are loaded first, and every new game is
        appended as soon as it ends, so an interrupted run can simply be restarted.
        """
        if csv_path and os.path.exists(csv_path) and not self.results:
            self.results = load_results(csv_path)
        done = {(r["agent"], r["seed"]) for r in self.results}
        tasks = [(n, f, s) for n, f in self.agents.items() for s in self.test_seeds
                 if (n, s) not in done]
        finished = 0

        def handle(record):
            nonlocal finished
            finished += 1
            self.results.append(record)
            if csv_path:
                _append_csv(csv_path, record)
            if progress:
                progress(finished, len(tasks), record)

        if workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for record in pool.map(_run_task, tasks, chunksize=1):
                    handle(record)
        else:
            for task in tasks:
                handle(_run_task(task))
        return self.results

    def calculate_metrics(self) -> Dict[str, Dict]:
        """Summary statistics per agent (in the order the agents were given)."""
        names = list(dict.fromkeys(list(self.agents) + [r["agent"] for r in self.results]))
        metrics = {}
        for name in names:
            rs = [r for r in self.results if r["agent"] == name]
            if not rs:
                continue
            scores = [r["final_score"] for r in rs]
            total_moves = max(sum(r["moves"] for r in rs), 1)
            tiles: Dict[int, int] = {}
            for r in rs:
                tiles[r["highest_tile"]] = tiles.get(r["highest_tile"], 0) + 1
            metrics[name] = {
                "games": len(rs),
                "avg_score": statistics.mean(scores),
                "std_score": statistics.pstdev(scores) if len(rs) > 1 else 0.0,
                "avg_highest_tile": statistics.mean(r["highest_tile"] for r in rs),
                "pct_reached_2048": 100 * sum(r["reached_2048"] for r in rs) / len(rs),
                "avg_moves": statistics.mean(r["moves"] for r in rs),
                "avg_decision_ms": 1000 * sum(r["decision_time_s"] for r in rs) / total_moves,
                "avg_nodes_per_move": sum(r["nodes_expanded"] for r in rs) / total_moves,
                "tile_counts": dict(sorted(tiles.items())),
            }
        return metrics

    def export_results(self, path: str) -> None:
        """Write every game to a CSV file."""
        _ensure_dir(path)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for r in self.results:
                writer.writerow({k: r[k] for k in FIELDS})

    def export_metrics(self, path: str) -> None:
        """Write the per-agent summary to a CSV file."""
        _ensure_dir(path)
        cols = ["agent", "games", "avg_score", "std_score", "avg_highest_tile",
                "pct_reached_2048", "avg_moves", "avg_decision_ms", "avg_nodes_per_move"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            for name, m in self.calculate_metrics().items():
                writer.writerow([name] + [round(m[c], 3) for c in cols[1:]])


def _ensure_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _append_csv(path: str, record: Dict) -> None:
    _ensure_dir(path)
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: record[k] for k in FIELDS})
