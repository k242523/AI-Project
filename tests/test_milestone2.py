import unittest
from ai2048 import Game2048, move_board
from ai2048.agents import ExpectimaxAgent
from ai2048.evaluator import HeuristicEvaluator

FULL_DEAD = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.ev = HeuristicEvaluator()

    def test_empty_cells(self):
        board = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 0]]
        self.assertEqual(self.ev.count_empty_cells(board), 14)

    def test_smoothness(self):
        smooth = [[2, 2, 0, 0]] + [[0] * 4] * 3
        rough = [[2, 64, 0, 0]] + [[0] * 4] * 3
        self.assertEqual(self.ev.calculate_smoothness(smooth), 0)
        self.assertLess(self.ev.calculate_smoothness(rough), 0)

    def test_monotonicity_prefers_ordered_rows(self):
        ordered = [[8, 4, 2, 0]] + [[0] * 4] * 3
        messy = [[2, 8, 0, 4]] + [[0] * 4] * 3
        self.assertGreater(self.ev.calculate_monotonicity(ordered),
                           self.ev.calculate_monotonicity(messy))

    def test_corner(self):
        corner = [[16, 0, 0, 0]] + [[0] * 4] * 3
        middle = [[0, 16, 0, 0]] + [[0] * 4] * 3
        self.assertGreater(self.ev.evaluate_corner_position(corner), 0)
        self.assertLess(self.ev.evaluate_corner_position(middle), 0)

    def test_evaluate_prefers_better_board(self):
        good = [[64, 32, 8, 2], [0] * 4, [0] * 4, [0] * 4]
        bad = [[2, 0, 0, 64], [0, 32, 0, 0], [8, 0, 4, 0], [0, 2, 0, 16]]
        self.assertGreater(self.ev.evaluate(good), self.ev.evaluate(bad))

    def test_custom_weights(self):
        ev = HeuristicEvaluator({"empty": 1.0, "smoothness": 0, "monotonicity": 0, "corner": 0})
        self.assertEqual(ev.evaluate([[0] * 4] * 4), 16.0)


class _CountFours:
    """Stub evaluator: utility = number of 4 tiles on the board."""
    def evaluate(self, board):
        return sum(v == 4 for row in board for v in row)


class TestExpectimax(unittest.TestCase):
    def test_chance_node_uses_90_10_probabilities(self):
        agent = ExpectimaxAgent(search_depth=1, evaluator=_CountFours())
        board = [[2, 8, 2, 8], [8, 2, 8, 2], [2, 8, 2, 8], [8, 2, 0, 0]]
        # Each empty cell gets a 4 with probability 0.1, so the expected count is 0.1
        self.assertAlmostEqual(agent.evaluate_chance_node(board, 1), 0.1)

    def test_board_not_modified_by_search(self):
        agent = ExpectimaxAgent(2)
        board = [[2, 4, 0, 0], [0, 2, 0, 0], [0, 0, 8, 0], [0, 0, 0, 2]]
        copy = [r[:] for r in board]
        agent.choose_move(board)
        self.assertEqual(board, copy)

    def test_only_legal_move_is_chosen(self):
        board = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 0]]
        legal = [d for d in ("up", "down", "left", "right") if move_board(board, d)[2]]
        self.assertIn(ExpectimaxAgent(2).choose_move(board), legal)

    def test_no_move_on_dead_board(self):
        self.assertIsNone(ExpectimaxAgent(2).choose_move(FULL_DEAD))

    def test_never_returns_illegal_move(self):
        game, agent = Game2048(seed=11), ExpectimaxAgent(1)
        for _ in range(60):
            if game.game_over:
                break
            self.assertTrue(game.move(agent.choose_move(game.board)))

    def test_deterministic(self):
        board = Game2048(seed=4).board
        self.assertEqual(ExpectimaxAgent(2).choose_move(board),
                         ExpectimaxAgent(2).choose_move(board))

    def test_nodes_counted_and_grow_with_depth(self):
        board = Game2048(seed=4).board
        a1, a2 = ExpectimaxAgent(1), ExpectimaxAgent(2)
        a1.choose_move(board)
        a2.choose_move(board)
        self.assertGreater(a1.nodes_expanded, 0)
        self.assertGreater(a2.nodes_expanded, a1.nodes_expanded)

    def test_chosen_move_is_not_a_certain_loss(self):
        from ai2048.agents import LOSS_VALUE
        board = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 2, 0]]
        agent = ExpectimaxAgent(2)
        move = agent.choose_move(board)
        after = move_board(board, move)[0]
        self.assertGreater(agent.evaluate_chance_node(after, 2), LOSS_VALUE / 2)


if __name__ == "__main__":
    unittest.main()
