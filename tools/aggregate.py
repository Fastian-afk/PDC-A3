#!/usr/bin/env python3
"""Imaad Fazal | 23i-0656 | Section C | Assignment 3.

Validate and deterministically aggregate actual raw measurements. This program
never supplies missing trials or converts rank-duration sums into wall time.
"""
import argparse
import csv
import math
import os
from pathlib import Path
import sys
import tempfile

SCHEMA = "schema_version,timestamp,run_id,experiment,question,version,mode,size,steps,processes,threads,nodes,repetition,step,record,rank,hostname,elapsed_s,compute_s,comm_s,wait_s,bytes_received,correct,stage0_s,stage1_s,stage2_s,stage3_s,vector_bytes,schedule,partition,wait_first".split(",")
GROUP = "run_id,experiment,question,version,mode,size,steps,processes,threads,nodes,step,vector_bytes,schedule,partition,wait_first".split(",")
TIMES = "elapsed_s,compute_s,comm_s,wait_s,stage0_s,stage1_s,stage2_s,stage3_s".split(",")
INTEGERS = "size,steps,processes,threads,nodes,repetition,step,rank,vector_bytes,wait_first".split(",")


def number(value, label, optional=False):
    if optional and value == "NA":
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{label}: invalid number {value!r}") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label}: expected finite nonnegative number, got {value!r}")
    return result


def validate(row, where):
    if None in row or any(v is None for v in row.values()):
        raise ValueError(f"{where}: wrong CSV field count")
    if row["schema_version"] != "1":
        raise ValueError(f"{where}: unsupported schema version")
    if row["correct"] != "PASS":
        raise ValueError(f"{where}: failed/unverified correctness; run must be investigated")
    for field in INTEGERS:
        try:
            int(row[field])
        except ValueError as exc:
            raise ValueError(f"{where}: invalid integer {field}") from exc
    for field in ("steps", "processes", "threads", "nodes"):
        if int(row[field]) < 1:
            raise ValueError(f"{where}: {field} must be positive")
    if int(row["size"]) < (0 if row["mode"] == "micro" else 1):
        raise ValueError(f"{where}: size must be positive (or zero for an empty-vector microbenchmark)")
    if int(row["nodes"]) > int(row["processes"]):
        raise ValueError(f"{where}: node count exceeds rank count")
    if int(row["repetition"]) < 0 or int(row["vector_bytes"]) < 0:
        raise ValueError(f"{where}: negative repetition or vector size")
    if int(row["step"]) < -1 or int(row["step"]) >= int(row["steps"]):
        raise ValueError(f"{where}: step outside [-1, steps)")
    if row["wait_first"] not in ("0", "1"):
        raise ValueError(f"{where}: invalid wait_first")
    if row["record"] not in ("rank", "wall"):
        raise ValueError(f"{where}: unknown record kind")
    rank = int(row["rank"])
    if (row["record"] == "wall" and rank != -1) or (
            row["record"] == "rank" and not 0 <= rank < int(row["processes"])):
        raise ValueError(f"{where}: rank disagrees with record kind")
    for field in TIMES:
        number(row[field], f"{where}:{field}", optional=field != "elapsed_s")
    if row["record"] == "rank":
        for field in ("compute_s", "comm_s"):
            number(row[field], f"{where}:{field}")
        if row["wait_s"] != "NA" and float(row["wait_s"]) > float(row["comm_s"]) + 1e-9:
            raise ValueError(f"{where}: wait_s must be included in comm_s")
    if row["bytes_received"] != "NA":
        try:
            if int(row["bytes_received"]) < 0:
                raise ValueError()
        except ValueError as exc:
            raise ValueError(f"{where}: invalid received-byte count") from exc


def key(row):
    return tuple(row[field] for field in GROUP)


def read_raw(root, excluded=(), deduplicate=False):
    excluded = {p.resolve() for p in excluded}
    paths = sorted(p for p in root.rglob("*.csv") if p.resolve() not in excluded)
    if not paths:
        raise ValueError(f"no CSV measurement files under {root}")
    records, seen, trials = [], {}, {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != SCHEMA:
                raise ValueError(f"{path}: unexpected CSV schema; expected schema_version=1 raw data")
            for row in reader:
                where = f"{path}:{reader.line_num}"
                validate(row, where)
                identity = key(row) + (row["repetition"], row["record"], row["rank"])
                if identity in seen:
                    previous = seen[identity]
                    same = all(row[field] == previous[field] for field in SCHEMA)
                    if not deduplicate or not same:
                        raise ValueError(f"{where}: duplicate measurement identity (first at "
                                         f"{previous['source_file']}:{previous['source_line']}); "
                                         "use distinct run IDs for independent runs")
                    print(f"deduplicated identical row: {where}", file=sys.stderr)
                    continue
                row["source_file"] = path.relative_to(root).as_posix()
                row["source_line"] = str(reader.line_num)
                seen[identity] = row
                records.append(row)
                trials.setdefault(key(row) + (row["repetition"],), []).append(row)
    if not records:
        raise ValueError("input contains headers but no measurements")
    for identity, rows in trials.items():
        processes = int(rows[0]["processes"])
        walls = [r for r in rows if r["record"] == "wall"]
        ranks = [r for r in rows if r["record"] == "rank"]
        if len(walls) != 1 or len(ranks) != processes:
            raise ValueError(f"incomplete trial {identity}: expected one wall and {processes} rank rows")
        measured = max(float(r["elapsed_s"]) for r in ranks)
        wall = float(walls[0]["elapsed_s"])
        if not math.isclose(measured, wall, rel_tol=2e-10, abs_tol=1e-12):
            raise ValueError(f"wall is not maximum rank elapsed time for trial {identity}")
    return sorted(records, key=lambda r: key(r) + (int(r["repetition"]), int(r["rank"])))


def best_rows(records, expected):
    groups = {}
    for row in records:
        if row["record"] == "wall":
            groups.setdefault(key(row), []).append(row)
    summaries = []
    for identity, rows in sorted(groups.items()):
        chosen = min(rows, key=lambda r: (float(r["elapsed_s"]), int(r["repetition"])))
        result = dict(zip(GROUP, identity))
        count = len(rows)
        result.update({
            "valid_repetitions": str(count), "expected_repetitions": str(expected),
            "complete5": "yes" if expected == 5 and count == 5 else "no",
            "complete_expected": "yes" if count == expected else "no",
            "selected_repetition": chosen["repetition"],
            "best_elapsed_s": chosen["elapsed_s"],
            "elapsed_per_step_s": format(float(chosen["elapsed_s"]) /
                (int(chosen["steps"]) if int(chosen["step"]) == -1 else 1), ".12g"),
            "source_file": chosen["source_file"], "source_line": chosen["source_line"],
        })
        summaries.append(result)
    return summaries


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=path.parent, delete=False) as f:
            temp_path = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument("--best", type=Path, default=Path("best.csv"))
    parser.add_argument("--expected-reps", type=int, default=5)
    parser.add_argument("--require-complete", action="store_true",
                        help="fail unless every measured configuration has expected repetitions")
    parser.add_argument("--deduplicate-identical", action="store_true",
                        help="explicitly discard byte-identical duplicate records; conflicting duplicates fail")
    args = parser.parse_args()
    try:
        if args.expected_reps < 1:
            raise ValueError("--expected-reps must be positive")
        if args.output.resolve() == args.best.resolve():
            raise ValueError("--output and --best must differ")
        records = read_raw(args.raw_directory, (args.output, args.best), args.deduplicate_identical)
        summaries = best_rows(records, args.expected_reps)
        incomplete = sum(r["complete_expected"] != "yes" for r in summaries)
        if incomplete and args.require_complete:
            raise ValueError(f"{incomplete} configurations lack exactly {args.expected_reps} repetitions")
        write_csv(args.output, SCHEMA + ["source_file", "source_line"], records)
        write_csv(args.best, list(summaries[0]), summaries)
        print(f"validated {len(records)} raw records; {len(summaries)} measurement groups; "
              f"{incomplete} groups differ from {args.expected_reps} repetitions")
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"aggregation failed: {exc}\n")


if __name__ == "__main__":
    main()
