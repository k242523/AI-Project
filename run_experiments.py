"""Run the fixed-seed experiments from the proposal (Section 7).

Examples
  python run_experiments.py                         # 100 games for Random + depths 1, 2, 3
  python run_experiments.py --games 20 --depths 1 2 # quick test
  python run_experiments.py --depth-games 3=10      # depth 3 on only the first 10 seeds
  python run_experiments.py --workers 4             # use 4 CPU cores
  python run_experiments.py --charts-only           # redraw charts from results/games.csv

Results go to results/games.csv (one row per game), results/summary.csv and
results/charts/*.png. The run can be stopped and restarted: finished games are skipped.
"""
import argparse
import os
from functools import partial

from ai2048.agents import ExpectimaxAgent, RandomAgent
from ai2048.charts import make_charts
from ai2048.experiments import ExperimentRunner, load_results, make_seeds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=100, help="games per configuration")
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--depth-games", nargs="*", default=[], metavar="DEPTH=N",
                   help="play fewer games for slow depths, e.g. 3=10")
    p.add_argument("--workers", type=int, default=1, help="CPU processes to use")
    p.add_argument("--master-seed", type=int, default=2048)
    p.add_argument("--out", default="results")
    p.add_argument("--no-random", action="store_true")
    p.add_argument("--charts-only", action="store_true")
    args = p.parse_args()

    games_csv = os.path.join(args.out, "games.csv")
    seeds = make_seeds(args.games, args.master_seed)
    limits = {int(k): int(v) for k, v in (item.split("=") for item in args.depth_games)}

    if not args.charts_only:
        groups = []
        if not args.no_random:
            groups.append(({"Random": RandomAgent}, seeds))
        for d in args.depths:
            groups.append(({f"Expectimax-d{d}": partial(ExpectimaxAgent, d)}, seeds[:limits.get(d, args.games)]))

        results = None
        for agents, group_seeds in groups:
            name = next(iter(agents))
            print(f"\n{name}: {len(group_seeds)} games")

            def progress(done, total, rec, name=name):
                print(f"\r  {done}/{total}  seed {rec['seed']}  score {rec['final_score']:>6}  "
                      f"tile {rec['highest_tile']:>4}  {rec['avg_decision_ms']:.1f} ms/move   ", end="", flush=True)

            runner = ExperimentRunner(agents, group_seeds, results)
            results = runner.run_experiments(workers=args.workers, csv_path=games_csv, progress=progress)
            print()

    results = load_results(games_csv)
    runner = ExperimentRunner({}, [], results)
    runner.export_metrics(os.path.join(args.out, "summary.csv"))
    print("\n{:<16}{:>6}{:>12}{:>14}{:>10}{:>10}{:>12}".format(
        "Agent", "Games", "Avg score", "Avg max tile", "% 2048", "Moves", "ms/move"))
    for name, m in runner.calculate_metrics().items():
        print("{:<16}{:>6}{:>12,.0f}{:>14,.0f}{:>9.0f}%{:>10,.0f}{:>12.2f}".format(
            name, m["games"], m["avg_score"], m["avg_highest_tile"], m["pct_reached_2048"],
            m["avg_moves"], m["avg_decision_ms"]))
    for path in make_charts(results, os.path.join(args.out, "charts")):
        print("chart:", path)


if __name__ == "__main__":
    main()
