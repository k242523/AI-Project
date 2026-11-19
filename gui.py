"""Interactive 2048 with an Expectimax AI helper.

Run from the project folder:   python gui.py
Uses only Python's built-in Tkinter, so nothing needs to be installed.

Keys: arrows / WASD move | N new game | U undo | H hint | Space AI move | P auto-play | E experiments
"""
import os
import random
import threading
import time

try:
    import tkinter as tk
except ImportError:                                   # e.g. Linux without python3-tk
    raise SystemExit("Tkinter is missing. On Ubuntu/Debian run: sudo apt install python3-tk")

from ai2048 import Game2048, UP, DOWN, LEFT, RIGHT, move_board
from ai2048.agents import ExpectimaxAgent
from experiment_panel import ExperimentPanel

N, CELL, GAP = 4, 100, 12
BOARD_PX = N * CELL + (N + 1) * GAP
PAGE_BG, BOARD_BG, EMPTY_BG, DARK = "#faf8ef", "#bbada0", "#cdc1b4", "#776e65"
TILE_COLORS = {                      # value: (background, text)
    2: ("#eee4da", DARK), 4: ("#ede0c8", DARK), 8: ("#f2b179", "#f9f6f2"),
    16: ("#f59563", "#f9f6f2"), 32: ("#f67c5f", "#f9f6f2"), 64: ("#f65e3b", "#f9f6f2"),
    128: ("#edcf72", "#f9f6f2"), 256: ("#edcc61", "#f9f6f2"), 512: ("#edc850", "#f9f6f2"),
    1024: ("#edc53f", "#f9f6f2"), 2048: ("#edc22e", "#f9f6f2"),
}
BIG_TILE = ("#3c3a32", "#f9f6f2")    # anything above 2048
KEYMAP = {"up": UP, "down": DOWN, "left": LEFT, "right": RIGHT,
          "w": UP, "s": DOWN, "a": LEFT, "d": RIGHT}
ARROW = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}
BEST_FILE = os.path.join(os.path.expanduser("~"), ".ai2048_best")


class GameGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("2048 - Expectimax AI")
        root.configure(bg=PAGE_BG)
        root.resizable(False, False)

        self.game = Game2048(seed=random.randrange(2 ** 31))
        self.history = []                 # snapshots for Undo
        self.best = self._load_best()
        self.won = False
        self.busy = False                 # True while the AI is thinking in its thread
        self._token = 0                   # bumped to discard a stale AI answer
        self.hint_dir, self._hint_job = None, None
        self.banner = None                # (main text, small text) shown over the board
        self.auto = False
        self._auto_job = None
        self._pop_job = None
        self.pop_cell, self.pop_scale = None, 1.0

        self.depth = tk.IntVar(value=2)
        self.delay = tk.IntVar(value=150)
        self.adaptive = tk.BooleanVar(value=False)
        self.info = tk.StringVar(value="Arrow keys or WASD to play. Press Space to let the AI move.")
        self.score_var, self.best_var, self.max_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self._build()
        root.bind("<Key>", self.on_key)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.canvas.focus_set()
        self.refresh()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        top = tk.Frame(self.root, bg=PAGE_BG)
        top.pack(fill="x", padx=14, pady=(12, 2))
        tk.Label(top, text="2048", font=("Helvetica", 34, "bold"), fg=DARK, bg=PAGE_BG).pack(side="left")
        for title, var in (("MAX TILE", self.max_var), ("BEST", self.best_var), ("SCORE", self.score_var)):
            box = tk.Frame(top, bg=BOARD_BG, padx=10, pady=3)
            box.pack(side="right", padx=3)
            tk.Label(box, text=title, font=("Helvetica", 8, "bold"), fg="#eee4da", bg=BOARD_BG).pack()
            tk.Label(box, textvariable=var, font=("Helvetica", 14, "bold"), fg="white",
                     bg=BOARD_BG, width=7).pack()

        tk.Label(self.root, textvariable=self.info, bg=PAGE_BG, fg=DARK, font=("Helvetica", 10),
                 wraplength=BOARD_PX, justify="left", anchor="w", height=2).pack(fill="x", padx=14)

        self.canvas = tk.Canvas(self.root, width=BOARD_PX, height=BOARD_PX, bg=PAGE_BG,
                                highlightthickness=0, takefocus=0)
        self.canvas.pack(padx=14, pady=6)

        row1 = tk.Frame(self.root, bg=PAGE_BG)
        row1.pack(fill="x", padx=14, pady=(2, 4))
        self.auto_btn = None
        for text, cmd in (("New Game", self.new_game), ("Undo", self.undo), ("Hint", self.hint),
                          ("AI Move", self.ai_step), ("Auto-Play", self.toggle_auto)):
            b = tk.Button(row1, text=text, command=lambda c=cmd: self._click(c), takefocus=0,
                          font=("Helvetica", 10, "bold"), bg="#eee4da", fg="#3c3a32",
                          activebackground="#ddd0b8", relief="flat", bd=0, padx=6, pady=5)
            b.pack(side="left", expand=True, fill="x", padx=2)
            if text == "Auto-Play":
                self.auto_btn = b

        row2 = tk.Frame(self.root, bg=PAGE_BG)
        row2.pack(fill="x", padx=14, pady=2)
        tk.Label(row2, text="AI depth:", bg=PAGE_BG, fg=DARK).pack(side="left")
        for d in (1, 2, 3):
            tk.Radiobutton(row2, text=str(d), variable=self.depth, value=d, bg=PAGE_BG, fg=DARK,
                           activebackground=PAGE_BG, takefocus=0, command=self.canvas.focus_set
                           ).pack(side="left")
        tk.Label(row2, text="(3 is slow)", bg=PAGE_BG, fg="#a39a90", font=("Helvetica", 8)).pack(side="left")
        tk.Checkbutton(row2, text="Adaptive", variable=self.adaptive, bg=PAGE_BG, fg=DARK,
                       activebackground=PAGE_BG, takefocus=0, command=self.canvas.focus_set
                       ).pack(side="left", padx=(8, 0))
        tk.Scale(row2, from_=50, to=1000, orient="horizontal", variable=self.delay, length=150,
                 label="", showvalue=True, bg=PAGE_BG, highlightthickness=0, takefocus=0,
                 troughcolor="#eee4da").pack(side="right")
        tk.Label(row2, text="Auto delay (ms):", bg=PAGE_BG, fg=DARK).pack(side="right")

        row3 = tk.Frame(self.root, bg=PAGE_BG)
        row3.pack(fill="x", padx=14, pady=(2, 4))
        tk.Label(row3, text="Seed (blank = random):", bg=PAGE_BG, fg=DARK).pack(side="left")
        self.seed_entry = tk.Entry(row3, width=10)
        self.seed_entry.pack(side="left", padx=4)
        self.seed_entry.bind("<Return>", lambda e: self._click(self.new_game))
        self.seed_entry.bind("<Escape>", lambda e: self.canvas.focus_set())
        tk.Button(row3, text="Experiments...", command=lambda: self._click(self.open_experiments),
                  takefocus=0, font=("Helvetica", 10, "bold"), bg="#eee4da", fg="#3c3a32",
                  activebackground="#ddd0b8", relief="flat", bd=0, padx=8, pady=3).pack(side="right")

        tk.Label(self.root, text="Keys: arrows/WASD move · N new · U undo · H hint · Space AI move · P auto-play",
                 bg=PAGE_BG, fg="#a39a90", font=("Helvetica", 8)).pack(pady=(0, 10))

    def _click(self, command) -> None:
        command()
        self.canvas.focus_set()

    # ----------------------------------------------------------------- drawing
    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
               x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        return self.canvas.create_polygon(pts, smooth=True, **kw)

    def draw(self) -> None:
        c = self.canvas
        c.delete("all")
        self._round_rect(0, 0, BOARD_PX, BOARD_PX, 12, fill=BOARD_BG)
        for r in range(N):
            for col in range(N):
                x, y = GAP + col * (CELL + GAP), GAP + r * (CELL + GAP)
                v = self.game.board[r][col]
                scale = self.pop_scale if (r, col) == self.pop_cell else 1.0
                inset = CELL * (1 - scale) / 2
                bg, fg = TILE_COLORS.get(v, BIG_TILE) if v else (EMPTY_BG, DARK)
                self._round_rect(x + inset, y + inset, x + CELL - inset, y + CELL - inset, 8, fill=bg)
                if v and scale > 0.6:
                    size = 36 if v < 100 else 30 if v < 1000 else 24 if v < 10000 else 20
                    c.create_text(x + CELL / 2, y + CELL / 2, text=str(v), fill=fg,
                                  font=("Helvetica", size, "bold"))
        if self.hint_dir:
            m = BOARD_PX / 2
            dx, dy = {LEFT: (-1, 0), RIGHT: (1, 0), UP: (0, -1), DOWN: (0, 1)}[self.hint_dir]
            c.create_line(m - dx * 85, m - dy * 85, m + dx * 85, m + dy * 85, width=10, arrow="last",
                          arrowshape=(22, 26, 10), fill="#3a3d44", capstyle="round")
        if self.banner:
            main, sub = self.banner
            mid = BOARD_PX / 2
            self._round_rect(mid - 160, mid - 50, mid + 160, mid + 50, 14, fill=DARK)
            c.create_text(mid, mid - 14, text=main, fill="white", font=("Helvetica", 22, "bold"))
            c.create_text(mid, mid + 22, text=sub, fill="#eee4da", font=("Helvetica", 10))

    def refresh(self) -> None:
        g = self.game
        self.best = max(self.best, g.score)
        self.score_var.set(f"{g.score:,}")
        self.best_var.set(f"{self.best:,}")
        self.max_var.set(str(g.highest_tile))
        self.draw()

    def _pop(self, step: int = 0) -> None:
        scales = (0.45, 0.7, 0.88, 1.0)
        if step >= len(scales):
            self.pop_cell, self.pop_scale, self._pop_job = None, 1.0, None
        else:
            self.pop_scale = scales[step]
            self._pop_job = self.root.after(22, self._pop, step + 1)
        self.draw()

    # ------------------------------------------------------------------- moves
    def apply_move(self, direction: str) -> bool:
        if self.game.game_over:
            return False
        before = [list(r) for r in self.game.board]
        snap = self.game.snapshot()
        if not self.game.move(direction):
            self.info.set("That move changes nothing. Try another direction.")
            return False
        self.history.append(snap)
        self._clear_hint()
        slid = move_board(before, direction)[0]       # board before the new tile appeared
        self.pop_cell = next(((r, c) for r in range(N) for c in range(N)
                              if slid[r][c] == 0 and self.game.board[r][c]), None)
        if self._pop_job:
            self.root.after_cancel(self._pop_job)
        self.banner = None
        if self.game.game_over:
            self.banner = ("Game over!", "Press N for a new game or U to undo")
            self.stop_auto()
            self._save_best()
        elif self.game.highest_tile >= 2048 and not self.won:
            self.won = True
            self.banner = ("You reached 2048!", "Keep playing to get a higher score")
        self.refresh()
        self._pop()
        return True

    def _request_ai(self, then) -> None:
        """Run the Expectimax search in a thread and call then(move, ms, nodes, depth)
        on the Tk thread when it finishes. The window stays responsive meanwhile."""
        if self.busy:
            return
        self.busy = True
        self._token += 1
        token = self._token
        board = [list(r) for r in self.game.board]
        depth, adaptive = self.depth.get(), self.adaptive.get()
        box = {}

        def work():
            agent = ExpectimaxAgent(depth, adaptive_depth=adaptive)
            start = time.perf_counter()
            move = agent.choose_move(board)
            box["result"] = (move, (time.perf_counter() - start) * 1000, agent.nodes_expanded, depth)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        if depth >= 3 or not self.auto:
            self.info.set(f"AI thinking (depth {depth})...")
        self._poll_ai(thread, box, token, then)

    def _poll_ai(self, thread, box, token, then) -> None:
        if thread.is_alive():
            self.root.after(15, self._poll_ai, thread, box, token, then)
            return
        if token != self._token:           # cancelled (new game, undo, manual move, Stop)
            return
        self.busy = False
        then(*box["result"])

    def _cancel_ai(self) -> None:
        self._token += 1
        self.busy = False

    def ai_step(self) -> None:
        if self.game.game_over:
            self.stop_auto()
            return

        def done(move, ms, nodes, depth):
            if move is not None and self.apply_move(move):
                self.info.set(f"AI (depth {depth}) moved {ARROW[move]} · {nodes:,} nodes · {ms:.0f} ms")
            if self.auto and not self.game.game_over:
                self._auto_job = self.root.after(self.delay.get(), self.ai_step)

        self._request_ai(done)

    def hint(self) -> None:
        if self.game.game_over:
            return

        def done(move, ms, nodes, depth):
            self.hint_dir = move
            self.info.set(f"Hint: the AI (depth {depth}) would move {ARROW[move]} · {nodes:,} nodes · {ms:.0f} ms")
            self.draw()
            self._hint_job = self.root.after(1800, self._clear_hint)

        self._request_ai(done)

    def _clear_hint(self) -> None:
        if self._hint_job:
            self.root.after_cancel(self._hint_job)
            self._hint_job = None
        if self.hint_dir:
            self.hint_dir = None
            self.draw()

    def undo(self) -> None:
        self.stop_auto()
        if not self.history:
            self.info.set("Nothing to undo.")
            return
        self.game.restore(self.history.pop())
        self._clear_hint()
        self.banner, self.pop_cell = None, None
        self.info.set("Undid the last move.")
        self.refresh()

    def new_game(self) -> None:
        self.stop_auto()
        text = self.seed_entry.get().strip()
        try:
            seed = int(text) if text else random.randrange(2 ** 31)
        except ValueError:
            self.info.set("The seed must be a whole number (or leave it blank).")
            return
        self._save_best()
        self.game.reset(seed)
        self._clear_hint()
        self.history.clear()
        self.won, self.banner, self.pop_cell = False, None, None
        self.info.set(f"New game · seed {seed} (type it in the seed box to replay this game)")
        self.refresh()

    # --------------------------------------------------------------- auto-play
    def toggle_auto(self) -> None:
        if self.auto:
            self.stop_auto()
        elif not self.game.game_over:
            self.auto = True
            self.auto_btn.config(text="Stop")
            self.ai_step()

    def stop_auto(self) -> None:
        self.auto = False
        if self._auto_job:
            self.root.after_cancel(self._auto_job)
            self._auto_job = None
        self._cancel_ai()
        self.auto_btn.config(text="Auto-Play")

    # ------------------------------------------------------------------- input
    def on_key(self, event) -> None:
        if self.root.focus_get() is self.seed_entry:
            return
        key = event.keysym.lower()
        if key in KEYMAP:
            self.stop_auto()
            self.apply_move(KEYMAP[key])
        elif key == "n":
            self.new_game()
        elif key == "u":
            self.undo()
        elif key == "h":
            self.hint()
        elif key == "space":
            self.ai_step()
        elif key == "p":
            self.toggle_auto()
        elif key == "e":
            self.open_experiments()

    def open_experiments(self) -> None:
        panel = getattr(self, "_panel", None)
        if panel is not None and panel.win.winfo_exists():
            panel.win.lift()
        else:
            self._panel = ExperimentPanel(self.root)

    @staticmethod
    def _load_best() -> int:
        try:
            with open(BEST_FILE) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return 0

    def _save_best(self) -> None:
        self.best = max(self.best, self.game.score)
        try:
            with open(BEST_FILE, "w") as f:
                f.write(str(self.best))
        except OSError:
            pass

    def on_close(self) -> None:
        self._save_best()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    GameGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
