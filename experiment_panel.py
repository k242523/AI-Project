"""Experiment window for the GUI: run the fixed-seed comparison and see the results.

It plays the same games as run_experiments.py (same seeds), but with a small number
of games by default so it finishes quickly. The work runs in a background thread,
so the window never freezes.
"""
import os
import queue
import threading
import tkinter as tk
from functools import partial
from tkinter import messagebox, ttk

from ai2048.agents import ExpectimaxAgent, RandomAgent
from ai2048.experiments import ExperimentRunner, load_results, make_seeds

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PAGE_BG, DARK = "#faf8ef", "#776e65"
COLUMNS = (("agent", "Agent", 120), ("games", "Games", 55), ("avg_score", "Avg score", 80),
           ("avg_highest_tile", "Avg max tile", 90), ("pct_reached_2048", "% 2048", 60),
           ("avg_decision_ms", "ms / move", 75))
CONFIGS = (("Random", RandomAgent, True), ("Expectimax-d1", partial(ExpectimaxAgent, 1), True),
           ("Expectimax-d2", partial(ExpectimaxAgent, 2), True),
           ("Expectimax-d3", partial(ExpectimaxAgent, 3), False))


class _Stop(Exception):
    pass


class ExperimentPanel:
    def __init__(self, parent: tk.Misc):
        self.win = tk.Toplevel(parent)
        self.win.title("Experiments")
        self.win.configure(bg=PAGE_BG)
        self.win.resizable(False, False)
        self.runner = ExperimentRunner({}, [])
        self.queue: "queue.Queue" = queue.Queue()
        self.stop_flag = threading.Event()
        self.thread = None
        self.games = tk.StringVar(value="10")
        self.use = {name: tk.BooleanVar(value=on) for name, _, on in CONFIGS}

        tk.Label(self.win, bg=PAGE_BG, fg=DARK, justify="left", wraplength=520,
                 text="Every agent plays the same fixed seeds, so the comparison is fair. "
                      "Depth 3 is slow: use few games.").pack(anchor="w", padx=12, pady=(10, 4))
        opts = tk.Frame(self.win, bg=PAGE_BG)
        opts.pack(fill="x", padx=12)
        tk.Label(opts, text="Games per agent:", bg=PAGE_BG, fg=DARK).pack(side="left")
        tk.Entry(opts, textvariable=self.games, width=5).pack(side="left", padx=(4, 12))
        for name, _, _ in CONFIGS:
            tk.Checkbutton(opts, text=name, variable=self.use[name], bg=PAGE_BG, fg=DARK,
                           activebackground=PAGE_BG).pack(side="left")

        btns = tk.Frame(self.win, bg=PAGE_BG)
        btns.pack(fill="x", padx=12, pady=6)
        self.run_btn = self._button(btns, "Run", self.run)
        self.stop_btn = self._button(btns, "Stop", self.stop)
        self._button(btns, "Load saved results", self.load_saved)
        self._button(btns, "Save CSV", self.save_csv)
        self._button(btns, "Make charts", self.make_charts)

        self.bar = ttk.Progressbar(self.win, length=540, mode="determinate")
        self.bar.pack(padx=12)
        self.status = tk.StringVar(value="Choose agents and press Run.")
        tk.Label(self.win, textvariable=self.status, bg=PAGE_BG, fg=DARK, anchor="w").pack(fill="x", padx=12, pady=2)

        self.table = ttk.Treeview(self.win, columns=[c[0] for c in COLUMNS], show="headings", height=6)
        for key, title, width in COLUMNS:
            self.table.heading(key, text=title)
            self.table.column(key, width=width, anchor="w" if key == "agent" else "e")
        self.table.pack(padx=12, pady=(2, 12))

    def _button(self, parent, text, command):
        b = tk.Button(parent, text=text, command=command, bg="#eee4da", fg="#3c3a32",
                      activebackground="#ddd0b8", relief="flat", bd=0, padx=8, pady=4)
        b.pack(side="left", padx=2)
        return b

    # ----------------------------------------------------------------- running
    def run(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        try:
            n = int(self.games.get())
            if not 1 <= n <= 100:
                raise ValueError
        except ValueError:
            messagebox.showerror("Experiments", "Games per agent must be a whole number from 1 to 100.", parent=self.win)
            return
        agents = {name: factory for name, factory, _ in CONFIGS if self.use[name].get()}
        if not agents:
            messagebox.showerror("Experiments", "Tick at least one agent.", parent=self.win)
            return
        seeds = make_seeds(100, 2048)[:n]               # same first-n seeds as run_experiments.py
        self.runner = ExperimentRunner(agents, seeds)
        self.stop_flag.clear()
        self.queue = queue.Queue()
        runner, q = self.runner, self.queue

        def progress(done, total, record):
            q.put(("progress", done, total, record))
            if self.stop_flag.is_set():
                raise _Stop

        def work():
            try:
                runner.run_experiments(progress=progress)
                q.put(("done",))
            except _Stop:
                q.put(("stopped",))
            except Exception as exc:                    # show errors instead of dying silently
                q.put(("error", str(exc)))

        self.bar.config(value=0, maximum=n * len(agents))
        self.status.set("Running...")
        self.run_btn.config(state="disabled")
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()
        self._poll()

    def stop(self) -> None:
        self.stop_flag.set()

    def _poll(self) -> None:
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == "progress":
                    _, done, total, rec = msg
                    self.bar.config(maximum=total, value=done)
                    self.status.set(f"{done}/{total}  {rec['agent']}  seed {rec['seed']}  "
                                    f"score {rec['final_score']:,}  tile {rec['highest_tile']}")
                    self._fill_table()
                else:
                    self.status.set({"done": "Finished.", "stopped": "Stopped."}.get(msg[0], f"Error: {msg[-1]}"))
                    self.run_btn.config(state="normal")
                    self._fill_table()
        except queue.Empty:
            pass
        if self.thread and self.thread.is_alive():
            self.win.after(100, self._poll)
        elif not self.queue.empty():
            self.win.after(50, self._poll)

    def _fill_table(self) -> None:
        self.table.delete(*self.table.get_children())
        for name, m in self.runner.calculate_metrics().items():
            self.table.insert("", "end", values=(
                name, m["games"], f"{m['avg_score']:,.0f}", f"{m['avg_highest_tile']:,.0f}",
                f"{m['pct_reached_2048']:.0f}%", f"{m['avg_decision_ms']:.2f}"))

    # ----------------------------------------------------------- files / charts
    def load_saved(self) -> None:
        path = os.path.join(RESULTS_DIR, "games.csv")
        if not os.path.exists(path):
            self.status.set("No saved results yet. Run run_experiments.py first.")
            return
        self.runner = ExperimentRunner({}, [], load_results(path))
        self.status.set(f"Loaded {len(self.runner.results)} games from results/games.csv")
        self._fill_table()

    def save_csv(self) -> None:
        if not self.runner.results:
            self.status.set("Nothing to save yet.")
            return
        self.runner.export_results(os.path.join(RESULTS_DIR, "gui_games.csv"))
        self.runner.export_metrics(os.path.join(RESULTS_DIR, "gui_summary.csv"))
        self.status.set("Saved results/gui_games.csv and results/gui_summary.csv")

    def make_charts(self) -> None:
        if not self.runner.results:
            self.status.set("Run or load results first.")
            return
        try:
            from ai2048.charts import make_charts
            paths = make_charts(self.runner.results, os.path.join(RESULTS_DIR, "charts_gui"))
        except SystemExit as exc:                       # matplotlib missing
            self.status.set(str(exc))
            return
        self.status.set(f"Saved {len(paths)} charts in results/charts_gui/")
