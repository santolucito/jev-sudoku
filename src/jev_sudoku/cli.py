"""Command line entry point: `uv run jev-sudoku [puzzle] [options]`."""

from __future__ import annotations

import argparse
import os
import sys

from .lattice import Grid
from .oracle import JevNoulOracle, JevOracle, MockOracle
from .puzzles import load
from .solver import reference_solution, solve

PRICE_PER_MTOK = 0.042  # jev-1.13 input tokens; output is free


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Solve Sudoku by lattice deduction, with Jev as the deduction step.")
    ap.add_argument("puzzle", nargs="?", default="easy", help="built-in name (easy/medium/hard/all), 81-char string, or file")
    ap.add_argument(
        "--mode",
        choices=["jev", "hybrid", "code"],
        default="jev",
        help="jev: eliminations come only from Jev; hybrid: Jev + sound code propagation; code: no model",
    )
    ap.add_argument("--oracle", choices=["jev", "mock"], default="jev", help="mock runs offline, no API key")
    ap.add_argument(
        "--question",
        choices=["choice", "noul"],
        default="noul",
        help="noul: one yes/no per (cell, digit), like LDT's per-candidate head; choice: one Choice per cell",
    )
    ap.add_argument("--must", action="store_true", help="noul: also ask 'must d go here?' and pin cells Jev says are forced")
    ap.add_argument("--must-as", choices=["fact", "guess"], default="guess", help="apply 'must' answers as eliminations or as backtrackable guesses")
    ap.add_argument("--theta-must", type=float, default=0.5, help="pin a cell when its 'must' probability is above this")
    ap.add_argument("--noise", type=float, default=0.0, help="mock oracle: per-cell rate of random (possibly unsound) eliminations")
    ap.add_argument("--theta", type=float, default=None, help="eliminate a candidate when its probability is below this (default 0.05 noul, 0.02 choice)")
    ap.add_argument("--max-rounds", type=int, default=60, help="deduction rounds (oracle calls) before abstaining")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--batch-size", type=int, default=27, help="Choice questions per Jev request")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.theta is None:
        args.theta = 0.05 if args.question == "noul" else 0.02

    if args.mode != "code" and args.oracle == "jev" and not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set. Get one at https://console.typesafe.ai/keys, or use --oracle mock.", file=sys.stderr)
        return 2

    totals = {"solved": 0, "abstained": 0}
    for name, text in load(args.puzzle):
        puzzle = Grid.from_string(text)
        reference = reference_solution(puzzle)
        if args.mode == "code":
            oracle = None
        elif args.oracle == "mock":
            oracle = MockOracle(noise=args.noise)
        elif args.question == "noul":
            oracle = JevNoulOracle(model=args.model, must=args.must, theta_must=args.theta_must, must_as_guess=args.must_as == "guess")
        else:
            oracle = JevOracle(model=args.model, batch_size=args.batch_size)

        print(f"== {name} ({len(puzzle.open_cells())} empty cells, mode={args.mode}, oracle={args.oracle if oracle else '-'}/{args.question})")
        res = solve(
            puzzle,
            oracle,
            mode=args.mode,
            theta_elim=args.theta,
            max_rounds=args.max_rounds,
            reference=reference,
            verbose=args.verbose,
        )
        totals[res.status] += 1
        t = res.trace
        print(f"   {res.status.upper()} in {res.seconds:.1f}s")
        if res.grid is not None:
            print("   " + res.grid.pretty().replace("\n", "\n   "))
        print(
            f"   rounds={t.rounds} decisions={t.decisions} backtracks={t.backtracks} "
            f"oracle_eliminations={t.oracle_eliminations} unsound_eliminations={t.unsound_eliminations}"
        )
        if oracle is not None:
            s = oracle.stats
            cost = s.input_tokens / 1e6 * PRICE_PER_MTOK
            print(f"   requests={s.requests} questions={s.questions} input_tokens={s.input_tokens} (~${cost:.4f})")
    print(f"\nsolved {totals['solved']}, abstained {totals['abstained']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
