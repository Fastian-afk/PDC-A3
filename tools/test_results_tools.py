#!/usr/bin/env python3
"""Imaad Fazal | 23i-0656 | Section C | Assignment 3.

Regression tests use synthetic fixtures ONLY in temporary directories. Fixture
values are not benchmark measurements and are never shipped as result evidence.
"""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from aggregate import SCHEMA, best_rows, read_raw


def fixture():
    rows = []
    for repetition in range(5):
        for rank in (-1, 0, 1):
            row = dict.fromkeys(SCHEMA, "NA")
            row.update({"schema_version": "1", "timestamp": "2000-01-01T00:00:00Z",
                        "run_id": "SYNTHETIC_UNIT_TEST", "experiment": "unit-test",
                        "question": "Q1", "version": "C", "mode": "workload",
                        "size": "8", "steps": "1", "processes": "2", "threads": "1",
                        "nodes": "1", "repetition": str(repetition), "step": "-1",
                        "record": "wall" if rank == -1 else "rank", "rank": str(rank),
                        "hostname": "ALL" if rank == -1 else "fixture-only",
                        "elapsed_s": str(0.5 + repetition + (0.1 if rank != 0 else 0)),
                        "correct": "PASS", "vector_bytes": "0", "schedule": "static",
                        "partition": "rows", "wait_first": "0"})
            if rank >= 0:
                row.update({"compute_s": "0.1", "comm_s": "0.2", "wait_s": "0.05",
                            "bytes_received": "4096"})
            rows.append(row)
    return rows


def save(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(rows)


class ResultToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "raw.csv"
        self.rows = fixture()
        save(self.path, self.rows)

    def tearDown(self):
        self.temp.cleanup()

    def test_best_is_wall_minimum_and_five_distinct_repetitions(self):
        raw = read_raw(self.root)
        summaries = best_rows(raw, 5)
        self.assertEqual(len(raw), 15)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["complete5"], "yes")
        self.assertEqual(summaries[0]["selected_repetition"], "0")
        self.assertEqual(summaries[0]["best_elapsed_s"], "0.6")
        self.assertEqual(summaries[0]["source_file"], "raw.csv")

    def test_missing_rank_rejected(self):
        save(self.path, self.rows[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete trial"):
            read_raw(self.root)

    def test_failed_correctness_rejected(self):
        self.rows[2]["correct"] = "FAIL"
        save(self.path, self.rows)
        with self.assertRaisesRegex(ValueError, "failed/unverified"):
            read_raw(self.root)

    def test_nonfinite_and_negative_times_rejected(self):
        for value in ("nan", "inf", "-0.1"):
            self.rows[0]["elapsed_s"] = value
            save(self.path, self.rows)
            with self.assertRaisesRegex(ValueError, "finite nonnegative"):
                read_raw(self.root)

    def test_wrong_wall_rejected(self):
        self.rows[0]["elapsed_s"] = "0.7"
        save(self.path, self.rows)
        with self.assertRaisesRegex(ValueError, "maximum rank elapsed"):
            read_raw(self.root)

    def test_collective_exposed_wait_unavailable_is_preserved(self):
        for row in self.rows:
            row["wait_s"] = "NA"
        save(self.path, self.rows)
        records = read_raw(self.root)
        self.assertTrue(all(row["wait_s"] == "NA" for row in records))

    def test_zero_length_vector_microbenchmark_is_valid(self):
        for row in self.rows:
            row.update({"mode": "micro", "size": "0", "vector_bytes": "0"})
        save(self.path, self.rows)
        self.assertEqual(len(read_raw(self.root)), 15)

    def test_duplicate_identical_is_explicit_not_silent(self):
        save(self.root / "duplicate.csv", self.rows)
        with self.assertRaisesRegex(ValueError, "duplicate measurement"):
            read_raw(self.root)
        self.assertEqual(len(read_raw(self.root, deduplicate=True)), 15)

    def test_header_only_not_ready(self):
        save(self.path, [])
        with self.assertRaisesRegex(ValueError, "no measurements"):
            read_raw(self.root)

    def test_whole_run_and_steps_never_merged(self):
        extra = [dict(row, steps="2", step="0") for row in self.rows]
        base = [dict(row, steps="2") for row in self.rows]
        save(self.path, base + extra)
        summary = best_rows(read_raw(self.root), 5)
        self.assertEqual(len(summary), 2)
        whole = next(r for r in summary if r["step"] == "-1")
        step = next(r for r in summary if r["step"] == "0")
        self.assertEqual(whole["elapsed_per_step_s"], "0.3")
        self.assertEqual(step["elapsed_per_step_s"], "0.6")

    def test_cli_repeatability_and_rank_svg(self):
        script = Path(__file__).with_name("aggregate.py")
        output, best = self.root / "results.csv", self.root / "best.csv"
        cmd = [sys.executable, str(script), str(self.root), "--output", str(output),
               "--best", str(best), "--require-complete"]
        subprocess.run(cmd, check=True, capture_output=True)
        first = output.read_bytes(), best.read_bytes()
        subprocess.run(cmd, check=True, capture_output=True)
        self.assertEqual(first, (output.read_bytes(), best.read_bytes()))
        plot = Path(__file__).with_name("plots.py")
        svg = self.root / "chart.svg"
        subprocess.run([sys.executable, str(plot), str(output), "--output", str(svg)],
                       check=True, capture_output=True)
        self.assertIn("MPI calls (includes wait)", svg.read_text())


if __name__ == "__main__":
    unittest.main()
