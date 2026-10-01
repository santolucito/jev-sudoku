"""The abstract domain from the LDT paper: A = {cells} -> P({1..9}).

Each cell holds a set of alive candidate digits, stored as a 9-bit mask
(bit d-1 set <=> digit d is alive). The order is set inclusion per cell:
lower is more precise. TOP = every digit alive everywhere; BOTTOM = some
cell has no alive candidate (the state concretizes to no solution).

The solver only ever moves *down* this lattice via `meet`, which is what
makes the deduction monotone regardless of what the oracle proposes.
"""

from __future__ import annotations

from dataclasses import dataclass

ALL = 0x1FF  # digits 1..9 alive
N = 81


def bit(d: int) -> int:
    return 1 << (d - 1)


def digits(mask: int) -> list[int]:
    return [d for d in range(1, 10) if mask & bit(d)]


def popcount(mask: int) -> int:
    return mask.bit_count()


def rc(i: int) -> tuple[int, int]:
    return divmod(i, 9)


def cell_name(i: int) -> str:
    r, c = rc(i)
    return f"r{r + 1}c{c + 1}"


def _units() -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    rows = [[r * 9 + c for c in range(9)] for r in range(9)]
    cols = [[r * 9 + c for r in range(9)] for c in range(9)]
    boxes = [
        [(br * 3 + dr) * 9 + bc * 3 + dc for dr in range(3) for dc in range(3)]
        for br in range(3)
        for bc in range(3)
    ]
    return rows, cols, boxes


ROWS, COLS, BOXES = _units()
UNITS = ROWS + COLS + BOXES
ROW_OF = [i // 9 for i in range(N)]
COL_OF = [i % 9 for i in range(N)]
BOX_OF = [(i // 27) * 3 + (i % 9) // 3 for i in range(N)]
PEERS = [
    sorted((set(ROWS[ROW_OF[i]]) | set(COLS[COL_OF[i]]) | set(BOXES[BOX_OF[i]])) - {i})
    for i in range(N)
]


@dataclass(frozen=True)
class Grid:
    """An element of the abstract domain. Immutable so search can keep snapshots."""

    cells: tuple[int, ...]

    @classmethod
    def top(cls) -> Grid:
        return cls((ALL,) * N)

    @classmethod
    def from_string(cls, s: str) -> Grid:
        s = "".join(ch for ch in s if ch in "0123456789.")
        if len(s) != N:
            raise ValueError(f"expected 81 cells, got {len(s)}")
        return cls(tuple(ALL if ch in "0." else bit(int(ch)) for ch in s))

    def meet(self, other: Grid) -> Grid:
        """Greatest lower bound: intersect candidate sets cell-wise."""
        return Grid(tuple(a & b for a, b in zip(self.cells, other.cells)))

    def eliminate(self, removals: dict[int, int]) -> Grid:
        """Meet with the grid that has `removals[i]` struck from cell i."""
        return Grid(tuple(m & ~removals.get(i, 0) for i, m in enumerate(self.cells)))

    def assign(self, i: int, d: int) -> Grid:
        cells = list(self.cells)
        cells[i] &= bit(d)
        return Grid(tuple(cells))

    def leq(self, other: Grid) -> bool:
        return all(a & ~b == 0 for a, b in zip(self.cells, other.cells))

    @property
    def is_bottom(self) -> bool:
        return any(m == 0 for m in self.cells)

    @property
    def is_fixed(self) -> bool:
        return all(popcount(m) == 1 for m in self.cells)

    def open_cells(self) -> list[int]:
        return [i for i, m in enumerate(self.cells) if popcount(m) > 1]

    def value(self, i: int) -> int | None:
        m = self.cells[i]
        return digits(m)[0] if popcount(m) == 1 else None

    def alive(self) -> int:
        return sum(popcount(m) for m in self.cells)

    def to_string(self) -> str:
        return "".join(str(self.value(i) or ".") for i in range(N))

    def pretty(self) -> str:
        out = []
        for r in range(9):
            if r and r % 3 == 0:
                out.append("------+-------+------")
            row = [str(self.value(r * 9 + c) or ".") for c in range(9)]
            out.append(" ".join(row[0:3]) + " | " + " ".join(row[3:6]) + " | " + " ".join(row[6:9]))
        return "\n".join(out)


def has_conflict(g: Grid) -> bool:
    """Sound conflict check (the paper's explicit conflict head, done exactly in code).

    BOTTOM, two fixed peers with the same digit, or a unit where some digit
    has nowhere left to go.
    """
    if g.is_bottom:
        return True
    for unit in UNITS:
        seen = 0
        union = 0
        for i in unit:
            m = g.cells[i]
            union |= m
            if popcount(m) == 1:
                if seen & m:
                    return True
                seen |= m
        if union != ALL:
            return True
    return False


def is_valid_solution(g: Grid, givens: Grid) -> bool:
    """Final gate before returning an answer: a full, consistent grid that respects the givens."""
    return g.is_fixed and g.leq(givens) and not has_conflict(g)


def sound_eliminations(g: Grid) -> dict[int, int]:
    """The one-step sound deduction operator for local constraints.

    Naked singles (a fixed digit is removed from its peers) and hidden
    singles (a digit with one home in a unit is fixed there). Every removal
    here is implied by the rules, so it never discards a real solution.
    """
    removals: dict[int, int] = {}

    def strike(i: int, mask: int) -> None:
        mask &= g.cells[i]
        if mask:
            removals[i] = removals.get(i, 0) | mask

    for i, m in enumerate(g.cells):
        if popcount(m) == 1:
            for p in PEERS[i]:
                strike(p, m)
    for unit in UNITS:
        for d in range(1, 10):
            homes = [i for i in unit if g.cells[i] & bit(d)]
            if len(homes) == 1 and popcount(g.cells[homes[0]]) > 1:
                strike(homes[0], g.cells[homes[0]] & ~bit(d))
    return removals


def propagate(g: Grid) -> Grid:
    """Iterate `sound_eliminations` to a fixpoint (or BOTTOM)."""
    while not has_conflict(g):
        removals = sound_eliminations(g)
        if not removals:
            break
        g = g.eliminate(removals)
    return g
