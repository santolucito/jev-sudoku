"""The deduction/search loop from the LDT paper (Algorithms 1 and 2), driven by an oracle.

    repeat
      deduce:   strike candidates the oracle scores below theta_elim   (lattice meet)
      conflict: code checks for BOTTOM / clashes                        (sound)
      decide:   if stuck, pin the least-constrained cell, trying digits
                in the oracle's preferred order
    until solved or out of budget

The paper stops on conflict during training. At inference we backtrack to the
last decision instead, since a conflict there only refutes that decision.
Eliminations made above every decision can't be undone, so if the oracle
wrongly strikes a true digit there the search runs dry and the solver
*abstains*. It never returns an unverified grid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from .lattice import Grid, bit, digits, has_conflict, is_valid_solution, popcount, propagate
from .oracle import Beliefs, Oracle

Mode = Literal["jev", "hybrid", "code"]


@dataclass
class Trace:
    rounds: int = 0
    decisions: int = 0
    backtracks: int = 0
    oracle_eliminations: int = 0
    # Only counted while the current state still contains the reference solution,
    # so it measures real soundness failures rather than eliminations inside a doomed branch.
    unsound_eliminations: int = 0
    log: list[str] = field(default_factory=list)


@dataclass
class Result:
    status: Literal["solved", "abstained"]
    grid: Grid | None
    trace: Trace
    seconds: float


def deduce(g: Grid, oracle: Oracle, theta_elim: float, trace: Trace, reference: Grid | None) -> tuple[Grid, Beliefs]:
    """One oracle-driven deduction round. Returns the narrowed grid and the beliefs it used."""
    beliefs = oracle.beliefs(g)
    removals: dict[int, int] = {}
    for i, probs in beliefs.items():
        for d, p in probs.items():
            if p < theta_elim and g.cells[i] & bit(d):
                removals[i] = removals.get(i, 0) | bit(d)
    n = sum(popcount(m) for m in removals.values())
    trace.oracle_eliminations += n
    if reference is not None and reference.leq(g):
        trace.unsound_eliminations += sum(1 for i, m in removals.items() if m & reference.cells[i])
    return g.eliminate(removals), beliefs


def solve(
    puzzle: Grid,
    oracle: Oracle | None,
    mode: Mode = "jev",
    theta_elim: float = 0.02,
    max_rounds: int = 60,
    reference: Grid | None = None,
    verbose: bool = False,
) -> Result:
    t0 = time.perf_counter()
    trace = Trace()

    def note(msg: str) -> None:
        trace.log.append(msg)
        if verbose:
            print(msg, flush=True)

    # Each stack frame: a state to explore plus the decision alternatives left at its parent.
    stack: list[Grid] = [propagate(puzzle) if mode != "jev" else puzzle]
    while stack:
        g = stack.pop()
        while True:
            if has_conflict(g):
                trace.backtracks += 1
                note(f"  conflict -> backtrack ({len(stack)} alternatives left)")
                break
            if g.is_fixed:
                if is_valid_solution(g, puzzle):
                    note("  solved and verified")
                    return Result("solved", g, trace, time.perf_counter() - t0)
                trace.backtracks += 1
                break
            before = g
            beliefs: Beliefs = {}
            if mode != "code":
                if trace.rounds >= max_rounds:
                    note(f"  round budget ({max_rounds}) exhausted -> abstain")
                    return Result("abstained", None, trace, time.perf_counter() - t0)
                trace.rounds += 1
                g, beliefs = deduce(g, oracle, theta_elim, trace, reference)
                note(f"round {trace.rounds}: alive {before.alive()} -> {g.alive()}, open cells {len(g.open_cells())}")
            if mode != "jev":
                g = propagate(g)
            # Oracle "must" answers taken as guesses, not facts: pin each one, leaving
            # "that digit is not here" on the stack so a wrong pin can be backed out.
            hints = [(i, d) for i, d, _ in getattr(oracle, "last_forced", []) if popcount(g.cells[i]) > 1 and g.cells[i] & bit(d)]
            for i, d in hints:
                if popcount(g.cells[i]) > 1 and g.cells[i] & bit(d):
                    stack.append(g.eliminate({i: bit(d)}))
                    g = g.assign(i, d)
                    trace.decisions += 1
            if hints:
                note(f"  guessed {len(hints)} forced cells from 'must' answers")
            if g != before:
                continue
            # Stuck: decide. Most-constrained cell; digits tried in oracle-preferred order.
            i = min(g.open_cells(), key=lambda j: popcount(g.cells[j]))
            order = digits(g.cells[i])
            if beliefs:
                probs = beliefs.get(i, {})
                order.sort(key=lambda d: probs.get(d, 0.0))  # ascending: best is pushed last, popped first
            trace.decisions += 1
            note(f"  decide on cell {i} among {order[::-1]}")
            stack.extend(g.assign(i, d) for d in order)
            break
    note("  search exhausted -> abstain")
    return Result("abstained", None, trace, time.perf_counter() - t0)


def reference_solution(puzzle: Grid) -> Grid | None:
    """Exact solution via pure code search, used for scoring runs only."""
    res = solve(puzzle, oracle=None, mode="code")
    return res.grid
