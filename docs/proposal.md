# AL-2002 Artificial Intelligence (Laboratory) · Fall 2026

**FAST National University of Computer and Emerging Sciences**
FAST School of Computing · Karachi Campus · Department of Artificial Intelligence & Data Science

## PROJECT PROPOSAL

Semester Lab Project · Weightage 1 mark

## Group Information

| | Name | Roll Number | Section | Role in the project |
|---|---|---|---|---|
| **Leader** | Shuzain Ali | 24K-2523 | BDS-5A | Everything. |
| Member 2 | | | | |

| | |
|---|---|
| **Project Title** | 2048 AI with Expectimax |
| **Project Idea** | Idea No. 06 from the Project Ideas list ☒ · Own idea (needs approval) ☐ |
| **Category** | Part A — Game ☒ · Part B — Applied / Data ☐ |
| **GitHub Repository URL** | https://github.com/k242523/AI-Project |
| **Date of Submission** | 22-10-2026 |

## 1. Project Title and Summary

This project aims to develop an artificial intelligence agent that can play the 2048 puzzle game automatically and achieve a high score by combining numbered tiles. The game is played on a 4×4 grid, where the player moves tiles in four directions, and matching tiles merge into a tile with their combined value. Unlike a human player who chooses moves based on intuition, the AI examines possible moves and predicts their likely outcomes before selecting an action. It uses the Expectimax search algorithm to account for the random appearance of new tiles and a heuristic evaluation function to estimate the quality of future game states. The system will provide a playable interface, display the AI's moves and score, and allow users to compare AI performance across multiple games. This project demonstrates how artificial intelligence can make strategic decisions in an environment involving both choices and randomness.

## 2. Problem Statement and Motivation

The 2048 puzzle game presents a decision-making problem in which a player must combine numbered tiles to achieve higher values without filling the board completely. Although the rules are simple, selecting the best move becomes increasingly difficult as the board fills and the number of possible future states grows. Each move can lead to several possible outcomes because a new tile, usually a 2 or 4, appears randomly in an empty cell. A brute-force approach would need to examine an enormous number of possible move sequences and tile placements, making exhaustive search computationally expensive even for a relatively small 4×4 board. Human players generally rely on experience and strategies, such as keeping large tiles in a corner, but these methods do not guarantee good decisions in every situation. The challenge is to build an agent that considers future possibilities, accounts for randomness, and selects moves efficiently. The Expectimax algorithm addresses this challenge by evaluating decisions and calculating expected outcomes at chance nodes. A heuristic function further estimates board quality without exploring every possible future state. This project is worthwhile because it demonstrates how search and probabilistic reasoning can improve automated decision-making in a game with a large search space.

## 3. Problem Formulation

### 3.1 PEAS Description

| Component | Description for your project |
|---|---|
| Performance measure | Achieve a high score, create large tiles, reach the 2048 tile, avoid game-over states, and complete moves within reasonable computation time. |
| Environment | The 2048 puzzle game played on a 4×4 grid, where tiles move and merge, and new tiles appear randomly in empty cells. |
| Actuators | Execute one of four actions: move up, move down, move left, or move right. |
| Sensors | Observe the current board configuration, tile values, empty cells, available legal moves, current score, and game-over status. |

### 3.2 Environment and Agent Type

| Property | Classification |
|---|---|
| Observability | Fully observable ☒ · Partially observable ☐ |
| Determinism | Deterministic ☐ · Stochastic ☒ |
| Change over time | Static ☒ · Dynamic ☐ |
| State and actions | Discrete ☒ · Continuous ☐ |
| Agents | Single-agent ☒ · Multi-agent / adversarial ☐ |
| Agent type | Simple reflex ☐ · Model-based ☐ · Goal-based ☐ · Utility-based ☒ · Learning ☐ |

### 3.3 Problem Representation

| Element | Your representation |
|---|---|
| State representation | A 4×4 matrix containing the value of each tile. An empty cell is represented by 0. The current score and other relevant game information may also be stored. |
| Actions / moves | Move Up, Move Down, Move Left, and Move Right. A move is legal if it changes the board by shifting or merging tiles. |
| Initial state | A 4×4 empty grid with two initial tiles placed in randomly selected empty cells, following the game's initialization rules. |
| Goal or terminal test | The main objective is to create a tile with value 2048 while keeping the game playable. The game terminates when no legal moves remain. For evaluation, the AI may continue playing after reaching 2048 to maximize its score or largest tile. |
| Cost, utility, fitness or evaluation | Expectimax estimates the expected value of each possible move. A heuristic function can reward empty cells, smooth tile arrangements, high scores, and favorable tile ordering, while penalizing poor arrangements and loss of mobility. The AI selects the legal move with the highest estimated expected utility. |

## 4. Proposed AI Approach

| # | Technique | Role in the project | Own code or library |
|---|---|---|---|
| 1 | Expectimax Search | Explores possible player moves and random tile placements, then chooses the move with the highest expected utility. | Own code |
| 2 | Heuristic Board Evaluation | Scores a board using empty cells, tile smoothness, monotonicity, and the largest tile position. | Own code |
| 3 | Depth-Limited Search Comparison | Tests different search depths, such as depth 2 and depth 3 or 4, to measure the trade-off between playing performance and computation time. | Own code |
| B | Baseline | Selects a legal move randomly, providing a simple reference against which to measure the Expectimax agent. | Own code |

Expectimax is suitable for 2048 because the agent controls the movement of tiles, but the appearance of new tiles is random. Unlike Minimax, which models an opponent choosing adverse actions, Expectimax calculates expected values for chance outcomes. A depth limit makes the search computationally manageable, while a heuristic evaluates board states that have not reached a terminal state. The heuristic will reward empty cells, smooth neighboring tile values, monotonic tile ordering, and keeping the largest tile in a corner. These properties help preserve space and make future merges easier. The agent will be compared with a random legal-move baseline using identical initial boards and random seeds. Different search depths will also be tested to determine whether deeper reasoning improves scores enough to justify its additional computational cost.

### 4.1 Planned Class Design

The project will use the following Python classes. Each class has a separate responsibility to keep the code organized and easy to test.

```python
class Game2048:
    # Attributes: board, score, game_over, highest_tile
    # Methods: move(direction), add_random_tile(),
    #          get_legal_moves(), is_game_over(),
    #          reset(seed), get_state()

class HeuristicEvaluator:
    # Attributes: heuristic_weights
    # Methods: evaluate(board), count_empty_cells(),
    #          calculate_smoothness(), calculate_monotonicity(),
    #          evaluate_corner_position()

class ExpectimaxAgent:
    # Attributes: search_depth, evaluator, nodes_expanded
    # Methods: choose_move(board), expectimax(),
    #          evaluate_chance_node(), evaluate_decision_node()

class RandomAgent:
    # Attributes: random_generator
    # Methods: choose_move(board)

class ExperimentRunner:
    # Attributes: agents, test_seeds, results
    # Methods: run_game(), run_experiments(),
    #          calculate_metrics(), export_results()
```

## 5. Scope

### In scope

| # | Feature | Why it is essential |
|---|---|---|
| 1 | Implement the complete 2048 game rules on a 4×4 grid. | Provides a correct environment for the AI. |
| 2 | Implement Expectimax with decision nodes, chance nodes, and configurable search depth. | This is the core AI technique being investigated. |
| 3 | Implement a heuristic evaluation function. | Allows the AI to estimate board quality beyond the search horizon. |
| 4 | Implement a random legal-move baseline and compare its performance with Expectimax. | Demonstrates whether the AI improves on a simple strategy. |
| 5 | Run reproducible experiments and display scores, highest tiles, and execution times through charts and a simple interface. | Provides evidence that the AI works and supports the final report. |

### Out of scope

The following features will deliberately not be included in the initial submission:

- Training a deep reinforcement-learning model.
- Multiplayer or online gameplay.
- A mobile application or online leaderboard.
- Advanced neural networks or GPU-based search.
- An exhaustive search of every possible future state.
- Comparing against numerous external AI implementations.

This scope prioritizes a working game, a correct Expectimax implementation, and reliable experiments over unnecessary features.

## 6. Data or Environment Source

| | |
|---|---|
| Source name and URL | Generated/simulated environment. Game rules can be cross-checked against the 2048 source repository |
| Type | Public dataset ☐ · Web API ☐ · Self-collected ☐ · Generated / simulated ☒ |
| Access requirements | No key needed ☒ · Free key required ☐ |
| Approximate size | One 4×4 board per game; 16 cells, with multiple initial states and random tile placements generated during play. |
| Licence / terms of use | The proposed Python implementation will be original code. If the reference game's source code or assets are reused, its MIT licence and attribution requirements will be checked and followed. |
| Offline fallback | No external data download is required. The game rules, AI code, test seeds, and experiment configuration will be stored locally, allowing all experiments to run without an internet connection. Results will be saved as local CSV files. |

## 7. Evaluation Plan

| | |
|---|---|
| What will be compared | Expectimax against RandomAgent, plus Expectimax at different search depths. |
| Metric(s) | Average final score, average highest tile, percentage of games reaching 2048, average number of moves, and average decision time per move. |
| Baseline | RandomAgent, which chooses uniformly from the available legal moves. |
| Test instances / test set | 100 predefined random seeds, with the same seeds used for every agent and configuration. |
| Number of runs and seed | 100 games per configuration. Use a fixed master seed, such as 2048, and record each game's seed. |
| Planned charts (at least three) | 1. Average final score by agent/configuration. 2. Highest-tile distribution by agent. 3. Average decision time versus search depth. 4. Percentage of games reaching 2048. |

## 8. Division of Work

| Member | Modules / components owned | Approximate share |
|---|---|---|
| Solo student | Game engine, board representation, tile movement, merging, random tile generation, and game-over detection | 25% |
| Solo student | Expectimax search, chance-node probabilities, heuristic evaluation, and search-depth configuration | 25% |
| Solo student | Random baseline, experiment runner, fixed-seed testing, metrics, and charts | 25% |
| Solo student | User interface, integration, debugging, documentation, and repository cleanup | 25% |

## 9. Milestones and Timeline

| # | Milestone — what will be demonstrably working | Target date | Owner |
|---|---|---|---|
| 1 | Complete the 2048 game engine with tile movement, merging, scoring, random tile generation, and game-over detection. | Week 08 | Solo Student |
| 2 | Implement Expectimax with decision nodes, chance nodes, configurable search depth, and heuristic evaluation. | Week 09 | Solo Student |
| 3 | Implement the RandomAgent baseline and test both agents on identical initial boards. | Week 10 | Solo Student |
| 4 | Run experiments at different search depths, collect performance metrics, and generate comparison charts. | Week 11-12 | Solo Student |
| 5 | Experiments complete, final integration, documentation and repository cleanup; prepare the final demonstration and report. | Week 13-14 | Solo Student |

## 10. Expected Outcome and Risks

The final submission will be a working 2048 game in which users can play manually or allow the AI to select moves automatically. The Expectimax agent will consider possible moves and random tile placements, using a heuristic evaluation function to choose promising actions. A random legal-move agent will provide a baseline for comparison. The experiment results will include average score, highest tile, success rate for reaching 2048, and computation time across fixed-seed games. Two main risks are excessive computation time at greater search depths and a heuristic that does not produce consistently strong play. If search becomes too slow, the search depth will be reduced or the evaluation process optimized. If performance is poor, heuristic weights will be tuned using separate development seeds before the final evaluation.
