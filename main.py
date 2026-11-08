"""Find a surviving grid path in O(n^2) time using four token states."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TextIO


DAMAGE, HEALING, PROTECTION, MULTIPLIER = range(4)
PROTECT_BIT, MULTIPLY_BIT = 1, 2
Tile = tuple[int, int]
Grid = list[list[Tile]]


@dataclass(frozen=True)
class Step:
    row: int
    col: int
    mental: int
    tokens: int
    action: str


def read_problem(stream: TextIO) -> tuple[Grid, int]:
    """Read n, M and exactly one record per non-starting tile."""
    lines = (line.split() for line in stream if line.strip())
    header = next(lines, [])
    if len(header) != 2:
        raise ValueError("The first line must contain n and M.")
    n, initial_mental = map(int, header)
    if not 1 <= n <= 100:
        raise ValueError("n must be between 1 and 100.")

    grid = [[(-1, 0) for _ in range(n)] for _ in range(n)]
    count = 0
    for fields in lines:
        if len(fields) != 4:
            raise ValueError("Each tile record must contain x y t v.")
        row, col, kind, value = map(int, fields)
        if not (0 <= row < n and 0 <= col < n) or (row, col) == (0, 0):
            raise ValueError("Tile coordinates must be in bounds and exclude (0, 0).")
        if grid[row][col][0] != -1:
            raise ValueError(f"Duplicate tile at ({row}, {col}).")
        if kind not in range(4) or value < 0:
            raise ValueError("Tile type must be 0..3 and value must be nonnegative.")
        if kind in (PROTECTION, MULTIPLIER) and value != 0:
            raise ValueError("Protection and multiplier tiles must have value 0.")
        grid[row][col] = (kind, value)
        count += 1
    if count != n * n - 1:
        raise ValueError(f"Expected {n * n - 1} tile records, received {count}.")
    return grid, initial_mental


def enter_tile(tile: Tile, mental: int, tokens: int):
    """Yield (remaining mental points, new token state, chosen action)."""
    kind, value = tile
    if kind == DAMAGE:
        if mental >= value:
            yield mental - value, tokens, "take_damage"
        if tokens & PROTECT_BIT:
            yield mental, tokens & ~PROTECT_BIT, "use_protection"
    elif kind == HEALING:
        yield mental + value, tokens, "heal"
        if tokens & MULTIPLY_BIT:
            yield mental + 2 * value, tokens & ~MULTIPLY_BIT, "use_multiplier"
    elif kind == PROTECTION:
        yield mental, tokens | PROTECT_BIT, "collect_protection"
    elif kind == MULTIPLIER:
        yield mental, tokens | MULTIPLY_BIT, "collect_multiplier"


def solve(grid: Grid, initial_mental: int) -> list[Step] | None:
    """Return one feasible path, including token actions, or None.

    best[row][col][tokens] stores the greatest remaining mental points for
    this cell and token inventory. -1 means unreachable. A lower mental
    value with the same inventory can never enable a better continuation.
    """
    if initial_mental < 0:
        return None
    n = len(grid)
    best = [[[-1] * 4 for _ in range(n)] for _ in range(n)]
    previous = [[[None] * 4 for _ in range(n)] for _ in range(n)]
    best[0][0][0] = initial_mental

    # Row-major order visits every predecessor before its successor.
    for row in range(n):
        for col in range(n):
            for tokens in range(4):
                mental = best[row][col][tokens]
                if mental < 0:
                    continue
                for next_row, next_col in ((row, col + 1), (row + 1, col)):
                    if next_row >= n or next_col >= n:
                        continue
                    for new_mental, new_tokens, action in enter_tile(
                        grid[next_row][next_col], mental, tokens
                    ):
                        if new_mental > best[next_row][next_col][new_tokens]:
                            best[next_row][next_col][new_tokens] = new_mental
                            previous[next_row][next_col][new_tokens] = (
                                row, col, tokens, action
                            )

    tokens = max(range(4), key=lambda state: best[n - 1][n - 1][state])
    if best[n - 1][n - 1][tokens] < 0:
        return None

    path = []
    row = col = n - 1
    while (row, col) != (0, 0):
        prev_row, prev_col, prev_tokens, action = previous[row][col][tokens]
        path.append(Step(row, col, best[row][col][tokens], tokens, action))
        row, col, tokens = prev_row, prev_col, prev_tokens
    path.append(Step(0, 0, initial_mental, 0, "start"))
    path.reverse()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="Input .txt file (default: stdin)")
    args = parser.parse_args()
    try:
        if args.input is None:
            grid, initial_mental = read_problem(sys.stdin)
        else:
            with args.input.open(encoding="utf-8") as stream:
                grid, initial_mental = read_problem(stream)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    path = solve(grid, initial_mental)
    if path is None:
        print("No path exists.")
        return
    print("Path found:")
    for step in path:
        print(f"({step.row}, {step.col}) mental={step.mental} action={step.action}")


if __name__ == "__main__":
    main()
