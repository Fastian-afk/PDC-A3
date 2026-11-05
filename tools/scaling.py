#!/usr/bin/env python3
"""Imaad Fazal | 23i-0656 | Section C | Assignment 3.

Compute speedup using an explicit baseline selector, with unchanged workload.
Selectors use JSON objects, for example --baseline '{"version":"omp",
"processes":"1","threads":"1","step":"-1"}'. Every selector must resolve
unambiguously. This avoids comparing a per-step time to a whole training run.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

from aggregate import write_csv


def selector(text):
    obj = json.loads(text)
    if not isinstance(obj, dict) or not obj:
        raise ValueError("selectors must be nonempty JSON objects")
    return {str(k): str(v) for k, v in obj.items()}


def matches(row, selected):
    unknown = selected.keys() - row.keys()
    if unknown:
        raise ValueError(f"unknown selector fields: {sorted(unknown)}")
    return all(row[k] == value for k, value in selected.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("best_csv", type=Path)
    parser.add_argument("--baseline", required=True, help="JSON object selecting one best.csv row")
    parser.add_argument("--targets", required=True, help="JSON object selecting compatible target rows")
    parser.add_argument("--output", type=Path, default=Path("scaling.csv"))
    parser.add_argument("--allow-incomplete", action="store_true", help="for smoke analysis only")
    args = parser.parse_args()
    try:
        baseline_selector, target_selector = selector(args.baseline), selector(args.targets)
        with args.best_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        baseline_rows = [r for r in rows if matches(r, baseline_selector)]
        if len(baseline_rows) != 1:
            raise ValueError(f"baseline selector resolves to {len(baseline_rows)} rows; require exactly one")
        baseline = baseline_rows[0]
        targets = [r for r in rows if matches(r, target_selector)]
        if not targets:
            raise ValueError("target selector resolves to no rows")
        for field in ("processes", "threads"):
            if int(baseline[field]) != 1:
                raise ValueError("efficiency baseline must use one process and one thread")
        workload = ("question", "mode", "size", "steps", "step", "vector_bytes")
        baseline_s = float(baseline["best_elapsed_s"])
        if not math.isfinite(baseline_s) or baseline_s <= 0:
            raise ValueError("baseline time must be positive and finite")
        output = []
        for row in targets:
            if any(row[field] != baseline[field] for field in workload):
                raise ValueError("target and baseline differ in question, mode, size, steps, step or vector_bytes")
            if not args.allow_incomplete and (
                    row["complete5"] != "yes" or baseline["complete5"] != "yes"):
                raise ValueError("scaling requires five trials; --allow-incomplete is for smoke analysis")
            elapsed = float(row["best_elapsed_s"])
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("target time must be positive and finite")
            resources = int(row["processes"]) * int(row["threads"])
            result = dict(row)
            result.update({"baseline_run_id": baseline["run_id"],
                           "baseline_version": baseline["version"],
                           "baseline_experiment": baseline["experiment"],
                           "baseline_source_file": baseline["source_file"],
                           "baseline_source_line": baseline["source_line"],
                           "baseline_elapsed_s": baseline["best_elapsed_s"],
                           "speedup": format(baseline_s / elapsed, ".12g"),
                           "efficiency": format(baseline_s / elapsed / resources, ".12g"),
                           "efficiency_denominator": "MPI_processes*OpenMP_threads"})
            output.append(result)
        output.sort(key=lambda r: (int(r["processes"]) * int(r["threads"]),
                                  r["version"], r["experiment"], r["run_id"]))
        write_csv(args.output, list(output[0]), output)
        print(f"wrote {len(output)} speedup/efficiency rows to {args.output}")
    except (ValueError, KeyError, OSError, csv.Error) as exc:
        parser.exit(1, f"scaling failed: {exc}\n")


if __name__ == "__main__":
    main()
