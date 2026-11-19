"""Let the Expectimax agent play one full game.
Example: python watch_ai.py --depth 2 --seed 2048 --show
"""
import argparse
import time

from ai2048 import Game2048
from ai2048.agents import ExpectimaxAgent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=2, help="search depth (player moves)")
    p.add_argument("--seed", type=int, default=2048)
    p.add_argument("--show", action="store_true", help="print the board after every move")
    args = p.parse_args()

    game, agent = Game2048(seed=args.seed), ExpectimaxAgent(args.depth)
    moves, start = 0, time.perf_counter()
    while not game.game_over:
        game.move(agent.choose_move(game.board))
        moves += 1
        if args.show:
            print(f"\nMove {moves}\n{game}")
    secs = time.perf_counter() - start
    print(f"\n{game}\nDepth {args.depth}, seed {args.seed}: {moves} moves, "
          f"{secs:.1f}s ({1000 * secs / moves:.0f} ms/move), "
          f"nodes expanded {agent.nodes_expanded}")


if __name__ == "__main__":
    main()
