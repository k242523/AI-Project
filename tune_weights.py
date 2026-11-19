"""Tune the heuristic weights on DEVELOPMENT seeds (Section 10 of the proposal).

Candidates are random rescalings of the hand-picked INITIAL_WEIGHTS.
Dev seeds are drawn from 2,000,000-3,000,000; the test seeds used by
run_experiments.py are below 1,000,000, so tuning never sees the test games.

  python tune_weights.py --candidates 40 --games 24
"""
import argparse
import csv
import json
import os
import random
import statistics
from concurrent.futures import ProcessPoolExecutor

from ai2048 import Game2048
from ai2048.agents import ExpectimaxAgent
from ai2048.evaluator import INITIAL_WEIGHTS as DEFAULT_WEIGHTS, HeuristicEvaluator
from ai2048.experiments import make_seeds

DEV_LOW, DEV_HIGH = 2_000_000, 3_000_000


def play_score(task) -> int:
    weights, seed, depth = task
    game = Game2048(seed=seed)
    agent = ExpectimaxAgent(depth, HeuristicEvaluator(weights))
    while not game.game_over:
        if not game.move(agent.choose_move(game.board)):
            break
    return game.score


def mean_score(pool, weights, seeds, depth):
    scores = list(pool.map(play_score, [(weights, s, depth) for s in seeds], chunksize=1))
    return statistics.mean(scores), statistics.pstdev(scores)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=int, default=40)
    p.add_argument("--games", type=int, default=24, help="dev games per candidate")
    p.add_argument("--depth", type=int, default=1, help="search depth while tuning (1 is fast)")
    p.add_argument("--validate-games", type=int, default=30)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    rng = random.Random(args.seed)
    tune_seeds = make_seeds(args.games, 111, DEV_LOW, DEV_HIGH)
    candidates = [dict(DEFAULT_WEIGHTS)] + [
        {k: round(v * 2 ** rng.uniform(-2, 2), 4) for k, v in DEFAULT_WEIGHTS.items()}
        for _ in range(args.candidates - 1)]

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, w in enumerate(candidates):
            mean, std = mean_score(pool, w, tune_seeds, args.depth)
            rows.append((mean, std, w))
            print(f"\rcandidate {i + 1}/{len(candidates)}  mean score {mean:8,.0f}", end="", flush=True)
        print()
        rows.sort(key=lambda r: -r[0])
        default_row = next(r for r in rows if r[2] == DEFAULT_WEIGHTS)
        best = rows[0]

        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "tuning.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["rank", "mean_score", "std_score"] + list(DEFAULT_WEIGHTS))
            for rank, (mean, std, w) in enumerate(rows, 1):
                wr.writerow([rank, round(mean, 1), round(std, 1)] + [w[k] for k in DEFAULT_WEIGHTS])

        print(f"\nOn the {args.games} tuning seeds (depth {args.depth}):")
        print(f"  default  {default_row[0]:9,.0f}  {DEFAULT_WEIGHTS}")
        print(f"  best     {best[0]:9,.0f}  {best[2]}")

        # fresh seeds the search never saw: guards against picking a lucky candidate
        val_seeds = make_seeds(args.validate_games, 222, DEV_LOW, DEV_HIGH)
        report = {"default": dict(DEFAULT_WEIGHTS), "best": best[2], "validation": {}}
        print(f"\nValidation on {args.validate_games} unseen dev seeds:")
        for depth, n in ((1, args.validate_games), (2, max(args.validate_games // 3, 5))):
            d_mean, _ = mean_score(pool, DEFAULT_WEIGHTS, val_seeds[:n], depth)
            b_mean, _ = mean_score(pool, best[2], val_seeds[:n], depth)
            report["validation"][f"depth{depth}"] = {"games": n, "default": d_mean, "best": b_mean}
            print(f"  depth {depth} ({n} games): default {d_mean:9,.0f}   tuned {b_mean:9,.0f}   "
                  f"({100 * (b_mean - d_mean) / d_mean:+.1f}%)")
    with open(os.path.join(args.out, "tuned_weights.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
