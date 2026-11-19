import random
import unittest
from ai2048 import Game2048, move_board, slide_row_left, UP, DOWN, LEFT, RIGHT


class TestSlideRow(unittest.TestCase):
    def test_cases(self):
        cases = [
            ([0, 0, 0, 0], [0, 0, 0, 0], 0),
            ([2, 0, 0, 2], [4, 0, 0, 0], 4),
            ([2, 2, 2, 2], [4, 4, 0, 0], 8),     # each tile merges only once
            ([2, 2, 4, 0], [4, 4, 0, 0], 4),
            ([4, 2, 2, 0], [4, 4, 0, 0], 4),
            ([2, 2, 2, 0], [4, 2, 0, 0], 4),     # leftmost pair merges first
            ([2, 4, 8, 16], [2, 4, 8, 16], 0),
            ([0, 0, 2, 4], [2, 4, 0, 0], 0),     # slide without merge
        ]
        for row, expected, score in cases:
            with self.subTest(row=row):
                self.assertEqual(slide_row_left(row), (expected, score))


class TestMoveBoard(unittest.TestCase):
    def setUp(self):
        self.board = [[2, 0, 0, 2],
                      [0, 4, 4, 0],
                      [0, 0, 0, 0],
                      [2, 0, 0, 0]]

    def test_left(self):
        nb, s, moved = move_board(self.board, LEFT)
        self.assertEqual(nb, [[4, 0, 0, 0], [8, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]])
        self.assertEqual(s, 12)
        self.assertTrue(moved)

    def test_right(self):
        nb, s, _ = move_board(self.board, RIGHT)
        self.assertEqual(nb, [[0, 0, 0, 4], [0, 0, 0, 8], [0, 0, 0, 0], [0, 0, 0, 2]])
        self.assertEqual(s, 12)

    def test_up(self):
        nb, s, _ = move_board(self.board, UP)
        self.assertEqual(nb, [[4, 4, 4, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(s, 4)

    def test_down(self):
        nb, s, _ = move_board(self.board, DOWN)
        self.assertEqual(nb, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 4, 4, 2]])
        self.assertEqual(s, 4)

    def test_input_not_mutated(self):
        copy = [r[:] for r in self.board]
        move_board(self.board, LEFT)
        self.assertEqual(self.board, copy)

    def test_no_change_is_not_a_move(self):
        board = [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.assertFalse(move_board(board, LEFT)[2])

    def test_invalid_direction(self):
        with self.assertRaises(ValueError):
            move_board(self.board, "diagonal")


class TestGame(unittest.TestCase):
    def test_initial_state(self):
        g = Game2048(seed=1)
        tiles = [v for row in g.board for v in row if v]
        self.assertEqual(len(tiles), 2)
        self.assertTrue(all(v in (2, 4) for v in tiles))
        self.assertEqual(g.score, 0)
        self.assertFalse(g.game_over)

    def test_same_seed_same_game(self):
        a, b = Game2048(seed=2048), Game2048(seed=2048)
        for d in [LEFT, UP, RIGHT, DOWN] * 10:
            self.assertEqual(a.move(d), b.move(d))
        self.assertEqual(a.get_state(), b.get_state())

    def test_reset_with_seed_reproduces_start(self):
        g = Game2048(seed=7)
        first = g.get_state()
        g.move(LEFT)
        g.move(UP)
        g.reset(seed=7)
        self.assertEqual(g.get_state(), first)

    def test_move_spawns_exactly_one_tile(self):
        g = Game2048(seed=3)
        g.board = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.assertTrue(g.move(RIGHT))
        self.assertEqual(sum(1 for r in g.board for v in r if v), 2)

    def test_illegal_move_changes_nothing(self):
        g = Game2048(seed=3)
        g.board = [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        before = g.get_state()
        self.assertFalse(g.move(LEFT))
        self.assertEqual(g.get_state(), before)

    def test_score_accumulates(self):
        g = Game2048(seed=3)
        g.board = [[2, 2, 4, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        g.move(LEFT)
        self.assertEqual(g.score, 12)
        self.assertEqual(g.board[0][:2], [4, 8])

    def test_highest_tile_updates(self):
        g = Game2048(seed=3)
        g.board = [[1024, 1024, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        g.move(LEFT)
        self.assertEqual(g.highest_tile, 2048)

    def test_game_over_detection(self):
        g = Game2048(seed=1)
        g.board = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        self.assertEqual(g.get_legal_moves(), [])
        self.assertTrue(g.is_game_over())

    def test_full_board_with_merge_is_not_over(self):
        g = Game2048(seed=1)
        g.board = [[2, 2, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        self.assertFalse(g.is_game_over())
        self.assertIn(LEFT, g.get_legal_moves())

    def test_no_moves_after_game_over(self):
        g = Game2048(seed=1)
        g.board = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        g.game_over = True
        self.assertFalse(g.move(LEFT))

    def test_get_state_returns_copy(self):
        g = Game2048(seed=1)
        state = g.get_state()
        state["board"][0][0] = 999
        self.assertNotEqual(g.board[0][0], 999)

    def test_spawn_probabilities(self):
        g = Game2048(seed=5)
        fours, n = 0, 4000
        for _ in range(n):
            g.board = [[0] * 4 for _ in range(4)]
            fours += g.add_random_tile()[2] == 4
        self.assertAlmostEqual(fours / n, 0.1, delta=0.02)

    def test_random_playouts_stay_valid(self):
        rng = random.Random(0)
        for seed in range(20):
            g = Game2048(seed=seed)
            while not g.game_over:
                self.assertTrue(g.move(rng.choice(g.get_legal_moves())))
                for row in g.board:
                    for v in row:
                        self.assertTrue(v == 0 or (v & (v - 1) == 0))  # power of 2
            self.assertEqual(g.get_legal_moves(), [])


class TestSnapshot(unittest.TestCase):
    def test_restore_returns_to_same_game_and_same_future(self):
        g = Game2048(seed=9)
        snap, state = g.snapshot(), g.get_state()
        g.move(LEFT)
        g.move(UP)
        g.restore(snap)
        self.assertEqual(g.get_state(), state)
        g.move(LEFT)                       # replaying gives the same tile spawn
        h = Game2048(seed=9)
        h.move(LEFT)
        self.assertEqual(g.get_state(), h.get_state())


if __name__ == "__main__":
    unittest.main()
