"""Independent path checks and exhaustive comparisons on small grids."""

import io
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest
from zipfile import ZipFile

from main import read_problem, solve


ROOT = Path(__file__).resolve().parents[1]


def exhaustive_possible(grid, initial):
    """Try every legal move and token choice without DP or dominance pruning."""
    n = len(grid)
    pending = [(0, 0, initial, False, False)]
    while pending:
        row, col, mental, protect, multiply = pending.pop()
        if mental < 0:
            continue
        if (row, col) == (n - 1, n - 1):
            return True
        for r, c in ((row + 1, col), (row, col + 1)):
            if r >= n or c >= n:
                continue
            kind, value = grid[r][c]
            if kind == 0:
                pending.append((r, c, mental - value, protect, multiply))
                if protect:
                    pending.append((r, c, mental, False, multiply))
            elif kind == 1:
                pending.append((r, c, mental + value, protect, multiply))
                if multiply:
                    pending.append((r, c, mental + 2 * value, protect, False))
            elif kind == 2:
                pending.append((r, c, mental, True, multiply))
            else:
                pending.append((r, c, mental, protect, True))
    return False


class SolverTests(unittest.TestCase):
    def assert_valid_path(self, grid, initial, path):
        """Replay the returned actions without calling the solver's transition code."""
        self.assertIsNotNone(path)
        n = len(grid)
        self.assertEqual(len(path), 2 * n - 1)
        self.assertEqual((path[0].row, path[0].col), (0, 0))
        self.assertEqual((path[-1].row, path[-1].col), (n - 1, n - 1))
        self.assertEqual((path[0].mental, path[0].tokens, path[0].action), (initial, 0, "start"))
        mental, tokens = initial, 0
        self.assertGreaterEqual(mental, 0)
        for before, after in zip(path, path[1:]):
            self.assertIn((after.row - before.row, after.col - before.col), ((1, 0), (0, 1)))
            kind, value = grid[after.row][after.col]
            if kind == 0:
                if after.action == "use_protection":
                    self.assertTrue(tokens & 1)
                    tokens &= ~1
                else:
                    self.assertEqual(after.action, "take_damage")
                    mental -= value
            elif kind == 1:
                if after.action == "use_multiplier":
                    self.assertTrue(tokens & 2)
                    tokens &= ~2
                    mental += 2 * value
                else:
                    self.assertEqual(after.action, "heal")
                    mental += value
            elif kind == 2:
                self.assertEqual(after.action, "collect_protection")
                tokens |= 1
            else:
                self.assertEqual(after.action, "collect_multiplier")
                tokens |= 2
            self.assertGreaterEqual(mental, 0)
            self.assertEqual((after.mental, after.tokens), (mental, tokens))

    @staticmethod
    def corridor(n, initial, tiles):
        """All routes except the top-row/right-column corridor are lethal."""
        grid = [[(0, 1000) for _ in range(n)] for _ in range(n)]
        grid[0][0] = (-1, 0)
        coordinates = [(0, c) for c in range(1, n)] + [(r, n - 1) for r in range(1, n)]
        for (r, c), tile in zip(coordinates, tiles):
            grid[r][c] = tile
        return grid, initial

    def test_basic_example_and_zero_boundary(self):
        with (ROOT / "examples" / "basic.txt").open() as stream:
            grid, mental = read_problem(stream)
        self.assert_valid_path(grid, mental, solve(grid, mental))
        damage = [[(-1, 0), (0, 2)], [(0, 20), (0, 3)]]
        path = solve(damage, 5)
        self.assert_valid_path(damage, 5, path)
        self.assertEqual(path[-1].mental, 0)
        self.assertIsNone(solve(damage, 4))

    def test_start_only_and_negative_start(self):
        grid, mental = read_problem(io.StringIO("1 0\n"))
        self.assert_valid_path(grid, mental, solve(grid, mental))
        self.assertIsNone(solve(grid, -1))
        healing = [[(-1, 0), (1, 100)], [(1, 100), (1, 100)]]
        self.assertIsNone(solve(healing, -1))

    def test_save_protection_for_larger_damage(self):
        grid, mental = self.corridor(3, 1, [(2, 0), (0, 1), (0, 100), (1, 0)])
        path = solve(grid, mental)
        self.assert_valid_path(grid, mental, path)
        self.assertEqual(path[2].action, "take_damage")
        self.assertEqual(path[3].action, "use_protection")

    def test_save_multiplier_for_larger_healing(self):
        grid, mental = self.corridor(3, 0, [(3, 0), (1, 1), (1, 10), (0, 21)])
        path = solve(grid, mental)
        self.assert_valid_path(grid, mental, path)
        self.assertEqual(path[2].action, "heal")
        self.assertEqual(path[3].action, "use_multiplier")

    def test_both_tokens_can_be_carried_and_used(self):
        grid, mental = self.corridor(
            4, 0, [(2, 0), (3, 0), (0, 100), (1, 10), (0, 19), (1, 0)]
        )
        path = solve(grid, mental)
        self.assert_valid_path(grid, mental, path)
        self.assertEqual(path[2].tokens, 3)
        self.assertIn("use_protection", [step.action for step in path])
        self.assertIn("use_multiplier", [step.action for step in path])

    def test_tokens_do_not_stack(self):
        for tiles in (
            [(2, 0), (2, 0), (0, 10), (0, 10)],
            [(3, 0), (3, 0), (1, 3), (0, 7)],
        ):
            grid, mental = self.corridor(3, 0, tiles)
            self.assertIsNone(solve(grid, mental))

    def test_random_small_grids_against_exhaustive_search(self):
        rng = random.Random(332)
        for case in range(400):
            n = rng.randrange(1, 5)
            grid = []
            for _ in range(n):
                row = []
                for _ in range(n):
                    kind = rng.randrange(4)
                    row.append((kind, rng.randrange(21) if kind < 2 else 0))
                grid.append(row)
            grid[0][0] = (-1, 0)
            initial = rng.randrange(-1, 16)
            with self.subTest(case=case):
                path = solve(grid, initial)
                self.assertEqual(path is not None, exhaustive_possible(grid, initial))
                if path is not None:
                    self.assert_valid_path(grid, initial, path)

    def test_maximum_grid_and_large_initial_mental(self):
        n = 100
        grid = [[(0, 1) for _ in range(n)] for _ in range(n)]
        grid[0][0] = (-1, 0)
        initial = 10 ** 100
        path = solve(grid, initial)
        self.assert_valid_path(grid, initial, path)
        self.assertEqual(path[-1].mental, initial - (2 * n - 2))

    def test_reject_malformed_input(self):
        bad_inputs = [
            "", "0 1\n", "101 1\n", "2 5\n", "1 0\n0 0 1 1\n",
            "2 1\n0 1 0 1\n0 1 0 1\n1 1 0 1\n",
            "2 1\n2 1 0 1\n", "2 1\n0 1 4 1\n",
            "2 1\n0 1 1 -1\n", "2 1\n0 1 2 3\n", "two five\n",
        ]
        for content in bad_inputs:
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_problem(io.StringIO(content))

    def test_cli_file_stdin_and_error(self):
        base = [sys.executable, str(ROOT / "main.py")]
        example = ROOT / "examples" / "basic.txt"
        by_file = subprocess.run(base + [str(example)], capture_output=True, text=True)
        by_stdin = subprocess.run(base, input=example.read_text(), capture_output=True, text=True)
        self.assertEqual(by_file.returncode, 0)
        self.assertEqual(by_stdin.returncode, 0)
        self.assertEqual(by_file.stdout, by_stdin.stdout)
        self.assertIn("(1, 1) mental=4", by_file.stdout)
        impossible = subprocess.run(base, input="1 -1\n", capture_output=True, text=True)
        self.assertEqual(impossible.returncode, 0)
        self.assertEqual(impossible.stdout.strip(), "No path exists.")
        malformed = subprocess.run(base, input="2 1\n", capture_output=True, text=True)
        self.assertEqual(malformed.returncode, 2)
        self.assertIn("Expected 3 tile records", malformed.stderr)

    @unittest.skipUnless(os.environ.get("COURSE_TEST_ZIP"), "Set COURSE_TEST_ZIP to test local course inputs.")
    def test_local_course_cases(self):
        with ZipFile(os.environ["COURSE_TEST_ZIP"]) as archive:
            names = [name for name in archive.namelist() if name.endswith(".txt")]
            self.assertTrue(names, "The archive contains no .txt test cases.")
            for name in names:
                with self.subTest(file=name):
                    folder = Path(name).parent.name
                    self.assertIn(folder, ("true", "false"))
                    grid, initial = read_problem(io.StringIO(archive.read(name).decode("utf-8")))
                    path = solve(grid, initial)
                    self.assertEqual(path is not None, folder == "true")
                    if path is not None:
                        self.assert_valid_path(grid, initial, path)


if __name__ == "__main__":
    unittest.main()
