import os
import random
import tempfile
import unittest
from functools import partial

from ai2048 import Game2048, move_board, move_tuple, slide_row_left, DIRECTIONS
from ai2048.agents import ExpectimaxAgent, RandomAgent
from ai2048.evaluator import HeuristicEvaluator
from ai2048.experiments import ExperimentRunner, make_seeds, load_results, play_game


class TestFastMoves(unittest.TestCase):
    def test_tuple_moves_match_reference_slide(self):
        rng = random.Random(1)
        for _ in range(300):
            board = [[rng.choice([0, 0, 2, 4, 8, 16]) for _ in range(4)] for _ in range(4)]
            for d in DIRECTIONS:
                new, gained, moved = move_board(board, d)
                tb = tuple(tuple(r) for r in board)
                ntb, g2, m2 = move_tuple(tb, d)
                self.assertEqual(new, [list(r) for r in ntb])
                self.assertEqual((gained, moved), (g2, m2))
        # left move equals slide_row_left applied to each row
        board = [[2, 2, 4, 0], [0, 0, 0, 0], [4, 4, 4, 4], [2, 0, 2, 8]]
        expected = [slide_row_left(r)[0] for r in board]
        self.assertEqual(move_board(board, "left")[0], expected)

    def test_evaluator_same_for_lists_and_tuples(self):
        ev = HeuristicEvaluator()
        board = [[2, 4, 8, 16], [0, 2, 0, 4], [0, 0, 0, 2], [0, 0, 0, 0]]
        self.assertAlmostEqual(ev.evaluate(board), ev.evaluate(tuple(map(tuple, board))))


class TestRandomAgent(unittest.TestCase):
    def test_always_legal(self):
        game, agent = Game2048(seed=3), RandomAgent(1)
        while not game.game_over:
            self.assertTrue(game.move(agent.choose_move(game.board)))

    def test_none_on_dead_board(self):
        dead = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        self.assertIsNone(RandomAgent(1).choose_move(dead))

    def test_reset_makes_games_reproducible(self):
        a, b = RandomAgent(), RandomAgent()
        self.assertEqual(play_game(a, 5)["final_score"], play_game(b, 5)["final_score"])
        self.assertEqual(play_game(a, 5)["final_score"], play_game(a, 5)["final_score"])


class TestAdaptiveDepth(unittest.TestCase):
    def test_deeper_only_when_crowded(self):
        agent = ExpectimaxAgent(2, adaptive_depth=True)
        crowded = tuple(tuple(r) for r in [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 0, 0]])
        roomy = tuple(tuple(r) for r in [[2, 0, 0, 0]] + [[0] * 4] * 3)
        self.assertEqual(agent._effective_depth(crowded), 3)
        self.assertEqual(agent._effective_depth(roomy), 2)
        self.assertEqual(ExpectimaxAgent(2)._effective_depth(crowded), 2)


class TestExperiments(unittest.TestCase):
    def setUp(self):
        self.seeds = make_seeds(4, master_seed=2048)
        self.agents = {"Random": RandomAgent, "Expectimax-d1": partial(ExpectimaxAgent, 1)}

    def test_make_seeds_fixed_and_unique(self):
        self.assertEqual(make_seeds(100), make_seeds(100))
        self.assertEqual(len(set(make_seeds(100))), 100)

    def test_same_seed_same_result(self):
        r = ExperimentRunner(self.agents, self.seeds)
        a, b = r.run_game("Expectimax-d1", self.seeds[0]), r.run_game("Expectimax-d1", self.seeds[0])
        self.assertEqual(a["final_score"], b["final_score"])
        self.assertEqual(a["moves"], b["moves"])

    def test_expectimax_beats_random_on_identical_seeds(self):
        runner = ExperimentRunner(self.agents, self.seeds)
        runner.run_experiments()
        m = runner.calculate_metrics()
        self.assertEqual(m["Random"]["games"], 4)
        self.assertGreater(m["Expectimax-d1"]["avg_score"], 3 * m["Random"]["avg_score"])
        self.assertGreater(m["Expectimax-d1"]["avg_highest_tile"], m["Random"]["avg_highest_tile"])

    def test_csv_export_resume_and_parallel(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "games.csv")
            first = ExperimentRunner(self.agents, self.seeds[:2])
            first.run_experiments(csv_path=path)
            self.assertEqual(len(load_results(path)), 4)
            # a new runner resumes: only the 2 missing seeds per agent are played
            second = ExperimentRunner(self.agents, self.seeds)
            second.run_experiments(workers=2, csv_path=path)
            self.assertEqual(len(second.results), 8)
            self.assertEqual(len(load_results(path)), 8)
            second.export_metrics(os.path.join(tmp, "summary.csv"))
            self.assertTrue(os.path.getsize(os.path.join(tmp, "summary.csv")) > 0)
            # parallel run gives the same numbers as the sequential one
            seq = ExperimentRunner(self.agents, self.seeds)
            seq.run_experiments()
            key = lambda r: (r["agent"], r["seed"])
            self.assertEqual([(r["agent"], r["seed"], r["final_score"]) for r in sorted(second.results, key=key)],
                             [(r["agent"], r["seed"], r["final_score"]) for r in sorted(seq.results, key=key)])


if __name__ == "__main__":
    unittest.main()
