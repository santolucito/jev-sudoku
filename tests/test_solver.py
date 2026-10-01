import pytest

from jev_sudoku.lattice import Grid, has_conflict, is_valid_solution, propagate
from jev_sudoku.oracle import MockOracle, cell_question, render_state
from jev_sudoku.puzzles import PUZZLES
from jev_sudoku.solver import reference_solution, solve


@pytest.mark.parametrize("name", list(PUZZLES))
def test_code_baseline_solves(name):
    puzzle = Grid.from_string(PUZZLES[name])
    res = solve(puzzle, None, mode="code")
    assert res.status == "solved"
    assert is_valid_solution(res.grid, puzzle)


def test_propagation_is_monotone():
    puzzle = Grid.from_string(PUZZLES["hard"])
    g = propagate(puzzle)
    assert g.leq(puzzle)
    assert not has_conflict(g)


@pytest.mark.parametrize("mode", ["jev", "hybrid"])
@pytest.mark.parametrize("name", list(PUZZLES))
def test_clean_mock_oracle_solves(mode, name):
    puzzle = Grid.from_string(PUZZLES[name])
    ref = reference_solution(puzzle)
    res = solve(puzzle, MockOracle(), mode=mode, max_rounds=2000, reference=ref)
    assert res.status == "solved"
    assert res.grid == ref
    assert res.trace.unsound_eliminations == 0


@pytest.mark.parametrize("seed", range(10))
def test_noisy_oracle_never_returns_wrong_grid(seed):
    puzzle = Grid.from_string(PUZZLES["medium"])
    ref = reference_solution(puzzle)
    res = solve(puzzle, MockOracle(noise=0.3, seed=seed), mode="jev", max_rounds=200, reference=ref)
    assert res.status in ("solved", "abstained")
    if res.status == "solved":
        assert res.grid == ref


def test_question_payload_shape():
    g = Grid.from_string(PUZZLES["easy"])
    i = g.open_cells()[0]
    q = cell_question(g, i)
    assert "row 1" in q and "column 3" in q and "box 1" in q
    assert len(render_state(g)["rows"]) == 9
