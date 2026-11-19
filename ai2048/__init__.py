"""2048 AI with Expectimax (AL-2002 semester project)."""
from .game import (Game2048, move_board, move_tuple, slide_row_left,
                   UP, DOWN, LEFT, RIGHT, DIRECTIONS)

__all__ = ["Game2048", "move_board", "move_tuple", "slide_row_left",
           "UP", "DOWN", "LEFT", "RIGHT", "DIRECTIONS"]
