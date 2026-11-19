# 2048 AI with Expectimax

AL-2002 Artificial Intelligence Lab, Fall 2026, FAST-NUCES Karachi.
Author: Shuzain Ali (24K-2523), BDS-5A.

An agent that plays 2048 using Expectimax search with a heuristic board
evaluation, compared against a random legal-move baseline.

## Progress

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Game engine (movement, merging, scoring, random tiles, game over) | Done |
| 2 | Expectimax + heuristic evaluation | Done |
| 3 | RandomAgent baseline, identical-seed tests | Done |
| 4 | Depth experiments, metrics, charts | Pipeline done; full 100-game runs to finish on the final machine |
| 5 | Integration, documentation, final report | In progress |

## Run

Requires Python 3.8+. Charts need `pip install matplotlib`.

```
python gui.py                          # visual game: play, undo, hint, watch the AI, run experiments
python play_cli.py                     # terminal version (w/a/s/d)
python watch_ai.py --depth 2 --seed 2048
python -m unittest discover tests      # run the tests

python tune_weights.py                 # tune heuristic weights on development seeds
python run_experiments.py              # 100 seeded games per agent -> results/
python run_experiments.py --workers 4  # use 4 CPU cores
python run_experiments.py --depth-games 3=10   # fewer games for slow depths
python run_experiments.py --charts-only        # redraw charts from results/games.csv
```

`run_experiments.py` saves every finished game to `results/games.csv`, so an
interrupted run can simply be started again and carries on where it stopped.

## Project structure

```
ai2048/game.py          Game2048 engine (+ fast cached move functions)
ai2048/evaluator.py     HeuristicEvaluator (empty cells, smoothness, monotonicity, corner)
ai2048/agents.py        ExpectimaxAgent (optional adaptive depth) and RandomAgent
ai2048/experiments.py   ExperimentRunner, fixed seeds, CSV export, metrics
ai2048/charts.py        the four evaluation charts
gui.py                  Tkinter window (AI runs in a background thread)
experiment_panel.py     experiment window opened from the GUI
run_experiments.py      command-line experiments
tune_weights.py         heuristic weight tuning
tests/                  unit tests
results/                CSV files and charts
docs/                   proposal and guides
```

## Method in short

* **Expectimax:** the player's move is a *max* node; the random new tile
  (a 2 with probability 0.9 or a 4 with 0.1, in a uniformly chosen empty cell)
  is a *chance* node that averages over every possibility.
  `search_depth` is the number of player moves looked ahead.
* **Heuristic** (used at the depth limit): rewards empty cells, smooth
  neighbouring tiles, monotonic rows and columns, and the largest tile in a corner.
* **Fair comparison:** every agent plays the same game seeds
  (`make_seeds(100, master_seed=2048)`); the random agent is re-seeded per game.
* **Tuning:** weights were tuned on development seeds (2,000,000 and above),
  which never overlap the test seeds (below 1,000,000).

## Preliminary results

Played on the proposal's fixed test seeds (tuned weights). The depth 2 and 3 rows
use fewer games because deep search is slow; the full 100-game runs are still to do.

| Agent | Games | Avg score | Avg highest tile | Reached 2048 | ms / move |
|---|---|---|---|---|---|
| Random | 100 | 1,073 | 105 | 0% | 0.01 |
| Expectimax depth 1 | 100 | 12,316 | 883 | 10% | 0.22 |
| Expectimax depth 2 | 20 | 25,678 | 1,766 | 65% | 5.18 |
| Expectimax depth 3 | 2 | 44,526 | 3,072 | 100% | 80.9 |

See `results/summary.csv` and `results/charts/`.
