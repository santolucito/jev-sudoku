"""Deduction oracles: given a lattice state, score every alive candidate.

In the LDT paper this is the recurrent transformer's elimination head,
sigma(b_ij). Here it is Jev: one Choice question per open cell, whose
per-option probabilities play the role of b_ij. The solver never trusts
these blindly -- it only uses them to propose moves down the lattice, and
code owns conflict detection and the final validity check.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Protocol

from .lattice import BOX_OF, BOXES, COL_OF, COLS, ROW_OF, ROWS, Grid, bit, cell_name, digits, popcount, sound_eliminations

# Per open cell: {digit: probability that this digit is the cell's value}.
Beliefs = dict[int, dict[int, float]]


@dataclass
class OracleStats:
    requests: int = 0
    questions: int = 0
    input_tokens: int = 0


class Oracle(Protocol):
    stats: OracleStats

    def beliefs(self, g: Grid) -> Beliefs: ...


def _render_cell(g: Grid, i: int) -> str:
    m = g.cells[i]
    if popcount(m) == 1:
        return str(digits(m)[0])
    return "{" + ",".join(map(str, digits(m))) + "}"


def render_state(g: Grid) -> dict:
    """The shared `state`: the whole board, fixed digits plus candidate sets."""
    rows = []
    for r in range(9):
        rows.append(" ".join(_render_cell(g, r * 9 + c) for c in range(9)))
    return {
        "puzzle": "9x9 Sudoku. Every row, every column and every 3x3 box must contain each digit 1-9 exactly once.",
        "notation": "A single digit is a placed value. {a,b,c} lists the digits still possible for an empty cell.",
        "rows": {f"row {r + 1}": rows[r] for r in range(9)},
    }


def cell_question(g: Grid, i: int) -> dict:
    """Instructions for one cell. The three units the cell belongs to are
    spelled out directly, so the model doesn't have to locate them in the
    state (the jev-1.13 docs flag indirection as a weak spot)."""
    r, c, b = ROW_OF[i], COL_OF[i], BOX_OF[i]

    def unit(cells: list[int]) -> dict[str, str]:
        return {cell_name(j) + (" (this cell)" if j == i else ""): _render_cell(g, j) for j in cells}

    return {
        "question": f"Which digit belongs in cell {cell_name(i)} (row {r + 1}, column {c + 1})?",
        "rules": [
            "A digit already placed anywhere in the same row, column or box cannot go in this cell.",
            "If a digit can only go in this cell within its row, column or box, it must go here.",
        ],
        f"row {r + 1}": unit(ROWS[r]),
        f"column {c + 1}": unit(COLS[c]),
        f"box {b + 1}": unit(BOXES[b]),
    }


class JevOracle:
    """Asks Jev one Choice per open cell, fanned out over a few concurrent requests."""

    def __init__(self, model: str = "jev-latest", batch_size: int = 27, concurrency: int = 4) -> None:
        from typesafe_sdk import AsyncTypeSafeClient  # imported lazily so offline modes need no key

        self._client_cls = AsyncTypeSafeClient
        self.model = model
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.stats = OracleStats()

    def beliefs(self, g: Grid) -> Beliefs:
        return asyncio.run(self._beliefs(g))

    async def _beliefs(self, g: Grid) -> Beliefs:
        from typesafe_sdk import Choice

        state = render_state(g)
        open_cells = g.open_cells()
        batches = [open_cells[k : k + self.batch_size] for k in range(0, len(open_cells), self.batch_size)]
        sem = asyncio.Semaphore(self.concurrency)
        out: Beliefs = {}

        async with self._client_cls(model=self.model) as client:

            async def run(batch: list[int]) -> None:
                questions = {
                    cell_name(i): Choice(
                        instructions=cell_question(g, i),
                        criteria={str(d): f"The digit {d} goes in {cell_name(i)}" for d in digits(g.cells[i])},
                    )
                    for i in batch
                }
                async with sem:
                    resp = await client.system_one(state=state, questions=questions)
                self.stats.requests += 1
                self.stats.questions += len(batch)
                self.stats.input_tokens += resp.usage.input_tokens or 0
                for i in batch:
                    probs = resp.choices[cell_name(i)].probabilities
                    out[i] = {int(k): float(v) for k, v in probs.items()}

            await asyncio.gather(*(run(b) for b in batches))
        return out


def _digit_status(g: Grid, j: int, d: int) -> str:
    m = g.cells[j]
    if m == bit(d):
        return f"holds {d}"
    if popcount(m) == 1:
        return "holds another digit"
    return f"empty, {d} possible" if m & bit(d) else f"empty, {d} ruled out"


def digit_state(g: Grid, d: int) -> dict:
    """The board as seen by one digit: where d is placed and where it can still go."""
    return {
        "puzzle": "9x9 Sudoku. Every row, every column and every 3x3 box must contain each digit 1-9 exactly once.",
        "digit": d,
        "rows": {f"row {r + 1}": [_digit_status(g, r * 9 + c, d) for c in range(9)] for r in range(9)},
    }


def digit_question(g: Grid, i: int, d: int) -> dict:
    """"Can d still go in cell i?" with the three units shown from d's point of view."""
    r, c, b = ROW_OF[i], COL_OF[i], BOX_OF[i]

    def unit(cells: list[int]) -> dict[str, str]:
        return {cell_name(j) + (" (this cell)" if j == i else ""): _digit_status(g, j, d) for j in cells}

    return {
        "question": f"Can the digit {d} go in cell {cell_name(i)} (row {r + 1}, column {c + 1})?",
        "digits still possible in this cell": digits(g.cells[i]),
        f"row {r + 1}": unit(ROWS[r]),
        f"column {c + 1}": unit(COLS[c]),
        f"box {b + 1}": unit(BOXES[b]),
    }


class JevNoulOracle:
    """The LDT elimination head, per candidate: one Noul per (open cell, alive digit).

    Each Noul is an independent yes/no like sigma(b_ij), not a share of a
    per-cell distribution, so a cell with many live digits doesn't dilute
    them. Requests are grouped by digit so each shares that digit's view of
    the board as `state`.

    With `must`, each candidate also gets "must d go here?" (d has no other
    home in some unit). A yes above `theta_must` pins the cell: every other
    digit there gets belief 0, which the solver then strikes. This is the
    cross-digit step the "can" question can't see.
    """

    def __init__(
        self,
        model: str = "jev-latest",
        batch_size: int = 60,
        concurrency: int = 6,
        must: bool = False,
        theta_must: float = 0.5,
        must_as_guess: bool = False,
    ) -> None:
        from typesafe_sdk import AsyncTypeSafeClient

        self._client_cls = AsyncTypeSafeClient
        self.model = model
        self.batch_size = batch_size // 2 if must else batch_size
        self.concurrency = concurrency
        self.must = must
        self.theta_must = theta_must
        self.must_as_guess = must_as_guess
        self.last_forced: list[tuple[int, int, float]] = []  # (cell, digit, p) for the solver to guess on
        self.stats = OracleStats()
        self.last_must: Beliefs = {}  # raw "must" probabilities from the last call, for analysis

    def beliefs(self, g: Grid) -> Beliefs:
        return asyncio.run(self._beliefs(g))

    async def _beliefs(self, g: Grid) -> Beliefs:
        from typesafe_sdk import Noul, NoulCriteria

        jobs: list[tuple[int, list[int]]] = []
        for d in range(1, 10):
            cells = [i for i in g.open_cells() if g.cells[i] & bit(d)]
            jobs += [(d, cells[k : k + self.batch_size]) for k in range(0, len(cells), self.batch_size)]
        sem = asyncio.Semaphore(self.concurrency)
        can: Beliefs = {i: {} for i in g.open_cells()}
        must: Beliefs = {i: {} for i in g.open_cells()}

        async with self._client_cls(model=self.model) as client:

            async def run(d: int, batch: list[int]) -> None:
                questions = {}
                for i in batch:
                    q = digit_question(g, i, d)
                    questions[cell_name(i)] = Noul(
                        instructions=q,
                        criteria=NoulCriteria(
                            true=f"No cell in the same row, column or box holds {d}, and this cell is not forced to hold a different digit",
                            false=f"Some cell in the same row, column or box already holds {d}, or this cell must hold a different digit",
                        ),
                    )
                    if self.must:
                        questions[cell_name(i) + "_must"] = Noul(
                            instructions={**q, "question": f"Must the digit {d} go in cell {cell_name(i)}?"},
                            criteria=NoulCriteria(
                                true=f"In this cell's row, column or box, this cell is the only cell where {d} is possible",
                                false=f"In each of this cell's row, column and box, {d} is possible in at least one other cell",
                            ),
                        )
                async with sem:
                    resp = await client.system_one(state=digit_state(g, d), questions=questions)
                self.stats.requests += 1
                self.stats.questions += len(questions)
                self.stats.input_tokens += resp.usage.input_tokens or 0
                for i in batch:
                    can[i][d] = float(resp.nouls[cell_name(i)].noul)
                    if self.must:
                        must[i][d] = float(resp.nouls[cell_name(i) + "_must"].noul)

            await asyncio.gather(*(run(d, b) for d, b in jobs))

        self.last_must = must
        if self.must_as_guess:
            self.last_forced = sorted(
                ((i, d, p) for i, probs in must.items() for d, p in probs.items() if p > self.theta_must),
                key=lambda t: -t[2],
            )
            return can
        for i, probs in must.items():
            forced = [d for d, p in probs.items() if p > self.theta_must]
            if len(forced) == 1:  # two forced digits in one cell is a contradiction; leave it to the "can" answers
                for d in can[i]:
                    if d != forced[0]:
                        can[i][d] = 0.0
        return can


@dataclass
class MockOracle:
    """Offline stand-in for Jev, for exercising the solver without an API key.

    Candidates that one step of sound deduction would strike get ~0; the rest
    share the mass. With `noise` > 0 it also, per cell, zeroes out a random
    alive candidate with that probability -- including sometimes the true
    one -- to simulate unsound eliminations and test that the solver abstains
    or recovers instead of returning a wrong grid.
    """

    noise: float = 0.0
    seed: int = 0
    stats: OracleStats = field(default_factory=OracleStats)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def beliefs(self, g: Grid) -> Beliefs:
        struck = sound_eliminations(g)
        out: Beliefs = {}
        for i in g.open_cells():
            alive = digits(g.cells[i])
            keep = [d for d in alive if not struck.get(i, 0) & bit(d)] or alive
            if self.noise and len(keep) > 1 and self._rng.random() < self.noise:
                keep.remove(self._rng.choice(keep))
            out[i] = {d: (1 / len(keep) if d in keep else 0.0) for d in alive}
        self.stats.requests += 1
        self.stats.questions += len(out)
        return out
