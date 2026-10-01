# jev-sudoku

Can a general-purpose model do the deduction step of
[Lattice Deduction Transformers](https://arxiv.org/abs/2605.08605) (LDT) **without the
Sudoku rules being written as code**?

This repo tries it with [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
TypeSafe's System One model. Jev answers typed questions (yes/no, multiple choice) with
calibrated probabilities. It is used zero-shot: no training on Sudoku, and the rules are
given only in plain English.

## TL;DR

- **Jev solved an easy puzzle with no wrong eliminations.** It ran the LDT loop with all
  eliminations coming from Jev. The setup that did this: one yes/no question per
  (cell, digit), plus "must" answers used as guesses the solver can back out of.
- **It did not solve medium or hard.** The guesses aren't good enough, so it backtracks too
  often and runs out of budget.
- **Asking the question the way LDT does mattered a lot.** One independent yes/no per
  candidate worked far better than one multiple-choice question per cell.
- **Jev reliably does "7 is already in this row, so it can't go here."** It's weak at
  "this cell must be a 4, so it can't be a 7."

## How it maps to LDT

| LDT | here |
| - | - |
| Abstract domain `A = cells -> P({1..9})` | `lattice.Grid`: one candidate set per cell |
| Per-candidate elimination head `sigma(b_ij) < theta_elim` | One Jev `Noul` (yes/no) per (open cell, alive digit): "can d go here?" Strike it if `p < --theta` |
| Projection through the lattice | `Grid.eliminate` is a meet, so the state only moves down |
| Decide: pin a cell, sample a digit | Pin the most-constrained cell, trying digits in Jev's order. With `--must`, also pin cells Jev says "must" hold a digit, as guesses |
| Conflict head | `has_conflict`, **in code** (see caveats) |
| "Correct answer or abstain" | A grid is returned only if it passes a final check. Otherwise the solver abstains |

Each question shows the cell's row, column and box from one digit's point of view
("holds 7", "empty, 7 possible", ...). Each request is grouped by digit, so the board view
is shared across questions.

## Findings

All runs used `jev-1.13.0`, zero-shot. "Unsound" counts eliminations that struck the true
digit while the solution was still reachable.

### 1. Question format: per-cell Choice vs per-candidate Noul

These numbers are from the first round on the easy puzzle. "Already-placed digits" means
digits that already appear in the cell's row, column or box.

| | Choice per cell ("which digit goes here?") | Noul per candidate ("can d go here?") |
| - | - | - |
| Median P(true digit) | 0.24 | – |
| Min P(true digit) | 0.16 (10th percentile) | **0.56** |
| Already-placed digits removed, safe threshold | ~half at 0.02 | **274/306 at 0.3** |
| True digits removed | 0 | 0 |

A Choice spreads its probability across the cell's candidates, so "unlikely to be *the*
answer" looks like "impossible". Nouls are independent per candidate, which matches LDT's
per-candidate sigmoid. Per-candidate Nouls are the default.

### 2. "Must" questions (forced cells)

After round 1, most of what's left to remove needs cross-digit reasoning: if r6c3 must be
a 4, it can't be a 5. Jev scored such removable digits as high as 0.88, because each
question only shows one digit. So I added "must d go in this cell?"

Detection on a fixed test position (no false alarms at any threshold):

| Threshold | easy: forced cells caught | medium: forced cells caught |
| - | - | - |
| 0.5 | 4/13 | 7/11 |
| 0.7 | 2/13 | 7/11 |
| 0.9 | 1/13 | 3/11 |

Used **as facts** (eliminate the other digits), one wrong call on easy and one on medium
removed the true digit and the runs abstained. Used **as backtrackable guesses**, there
were no unsound eliminations.

### 3. End-to-end results

| Setup | easy | medium | hard |
| - | - | - | - |
| Choice per cell, theta 0.02–0.12 | abstain (60 rounds) | – | – |
| Noul "can", theta 0.3 | abstain (60), 0 unsound | abstain (60), 0 unsound | abstain (60), 0 unsound |
| + "must" as facts | abstain, 1 unsound | abstain, 1 unsound | abstain (60) |
| + "must" as guesses | **solved in 69 rounds, 0 unsound, $0.25** | abstain (150), 47 backtracks, $0.64 | abstain (60), $0.55 |

For reference, the paper reports that frontier LLMs score 0% on its benchmarks
(Sudoku-Extreme and Snowflake Sudoku). LDT itself is an 800K-parameter model trained on
this exact loop and scores 100%.

### Caveats

- **Code still encodes the rules in two places:** the contradiction check, and the final
  answer check. In LDT the conflict check is learned. A stricter test would have Jev judge
  contradictions and keep only the final check in code.
- **The rules are stated in English in the questions**, e.g. "only cell in its row where 7
  is possible." That's within the spirit of "no formal rules," but it's more than just
  showing a Sudoku.
- **Small sample:** three puzzles, one run each. The thresholds were picked by looking at
  these same puzzles.
- `--mode code` is a classical hand-written solver. It's included as a sanity check, not as
  a fair baseline, since it is exactly the rule-writing this experiment tries to avoid.

## Ideas not tried

- Jev as the conflict detector (remove the rule check from code).
- Per-unit questions: "in row 3, which cell holds 7?" as a Choice over the open cells.
- Calibrating thresholds on a held-out set of puzzles, such as Sudoku-Extreme.

## Run it

```sh
uv sync
export TYPESAFE_API_KEY=...            # https://console.typesafe.ai/keys

uv run jev-sudoku easy --must --theta 0.3 --max-rounds 150 -v   # the setup that solved easy
uv run jev-sudoku all --question choice                          # per-cell Choice variant
uv run jev-sudoku medium --must --must-as fact                   # "must" as eliminations
uv run jev-sudoku hard --mode hybrid                             # Jev + code propagation

# offline, no key: a mock oracle, optionally noisy to test abstention
uv run jev-sudoku medium --oracle mock --noise 0.3
uv run pytest
```

Puzzles can be built-in names (`easy`, `medium`, `hard`, `all`), an 81-character string,
or a file with one puzzle per line (for a CSV, the first column is used).

| Flag | Meaning |
| - | - |
| `--mode jev / hybrid / code` | eliminations from Jev only / Jev plus code propagation / no model |
| `--question noul / choice` | one yes/no per (cell, digit), or one Choice per cell |
| `--theta` | strike a candidate when its probability is below this (default 0.05 noul, 0.02 choice) |
| `--must`, `--must-as guess / fact`, `--theta-must` | also ask "must d go here?" and how to use the answers |
| `--max-rounds` | deduction rounds before abstaining |

Code layout: `lattice.py` (the domain, the meet, the checks), `oracle.py` (Jev questions,
mock oracle), `solver.py` (deduce / decide / backtrack loop), `cli.py`.
