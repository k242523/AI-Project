"""Play 2048 in the terminal.  W/A/S/D to move, R to restart, Q to quit."""
from ai2048 import Game2048, UP, DOWN, LEFT, RIGHT

KEYS = {"w": UP, "s": DOWN, "a": LEFT, "d": RIGHT}


def main() -> None:
    game = Game2048()
    while True:
        print("\n" + str(game))
        if game.game_over:
            print("Game over! Press R to restart or Q to quit.")
        key = input("Move (w/a/s/d), r = restart, q = quit: ").strip().lower()
        if key == "q":
            break
        if key == "r":
            game.reset()
        elif key in KEYS:
            if not game.move(KEYS[key]) and not game.game_over:
                print("That move changes nothing, try another direction.")
        else:
            print("Use w, a, s, d, r or q.")


if __name__ == "__main__":
    main()
