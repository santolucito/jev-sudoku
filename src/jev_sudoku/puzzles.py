"""A few built-in puzzles, easy to hard. `.` or `0` marks an empty cell."""

PUZZLES: dict[str, str] = {
    # Solvable by naked/hidden singles alone.
    "easy": "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79",
    # Needs some deduction beyond singles, or a guess.
    "medium": "..9748...7.........2.1.9.....7...24..64.1.59..98...3.....8.3.2.........6...2759..",
    # Arto Inkala's 2012 "world's hardest sudoku".
    "hard": "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..",
}


def load(name_or_puzzle: str) -> list[tuple[str, str]]:
    """Resolve a built-in name, an 81-char puzzle, or a file with one puzzle per line."""
    if name_or_puzzle in PUZZLES:
        return [(name_or_puzzle, PUZZLES[name_or_puzzle])]
    if name_or_puzzle == "all":
        return list(PUZZLES.items())
    cleaned = "".join(ch for ch in name_or_puzzle if ch in "0123456789.")
    if len(cleaned) == 81:
        return [("puzzle", cleaned)]
    with open(name_or_puzzle) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    # Accept "puzzle" or "puzzle,solution,..." (e.g. CSV exports of Sudoku-Extreme).
    return [(f"line{k + 1}", ln.split(",")[0]) for k, ln in enumerate(lines)]
