#!/usr/bin/env python3
"""Imaad Fazal | 23i-0656 | Section C | Assignment 3.

Draw standalone SVG charts from actual validated CSVs, without dependencies.
--kind ranks selects one measurement configuration in results.csv using a JSON
--select object, chooses its minimum wall repetition, and draws separate rank
compute, MPI-call and wait bars. MPI-call bars include wait; bars are not stacked.
--kind scaling draws speedup and efficiency from scaling.py output.
"""
import argparse
import csv
from html import escape
import json
import math
from pathlib import Path

from aggregate import GROUP
from scaling import matches, selector


def svg_bars(title, subtitle, groups, series, ylabel, output):
    width = max(820, 125 * len(groups) + 200)
    height, left, top, plot_height = 540, 90, 100, 315
    right = width - 40
    maximum = max((value for _, values in series for value in values if value is not None), default=0)
    maximum = maximum * 1.1 if maximum else 1.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#17202a} .small{font-size:12px}</style>',
             f'<text x="{left}" y="30" font-size="19">{escape(title)}</text>',
             f'<text x="{left}" y="53" class="small">{escape(subtitle)}</text>',
             f'<text x="{left}" y="77" class="small">{escape(ylabel)}</text>']
    colors = ["#2563eb", "#e87918", "#059669"]
    for tick in range(6):
        y = top + plot_height * (1 - tick / 5)
        parts.append(f'<line x1="{left}" x2="{right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#d1d5db"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="small">{maximum * tick / 5:.4g}</text>')
    span = (right - left) / max(1, len(groups))
    bar_width = min(30, span * 0.75 / max(1, len(series)))
    for group_index, group in enumerate(groups):
        center = left + span * (group_index + 0.5)
        for series_index, (label, values) in enumerate(series):
            value = values[group_index]
            x = center + (series_index - len(series) / 2) * bar_width
            if value is None:
                parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{top + plot_height - 5}" text-anchor="middle" class="small">NA<title>{escape(str(group))}: {escape(label)} unavailable</title></text>')
                continue
            h = value / maximum * plot_height
            y = top + plot_height - h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 2:.2f}" height="{h:.2f}" fill="{colors[series_index % len(colors)]}"><title>{escape(str(group))}: {escape(label)}={value:.12g}</title></rect>')
        parts.append(f'<text x="{center:.2f}" y="{top + plot_height + 20}" text-anchor="middle" class="small">{escape(str(group))}</text>')
    for index, (label, _) in enumerate(series):
        x, y = left + index * 230, 470
        parts.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" fill="{colors[index % len(colors)]}"/>')
        parts.append(f'<text x="{x + 18}" y="{y}" class="small">{escape(label)}</text>')
    parts.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("chart requires finite nonnegative values")
    return number


def metric(value):
    return None if value == "NA" else positive(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--kind", choices=("ranks", "scaling"), default="ranks")
    parser.add_argument("--select", default="{}", help="JSON filter; must select one configuration for ranks")
    parser.add_argument("--output", type=Path, default=Path("chart.svg"))
    args = parser.parse_args()
    try:
        selected = json.loads(args.select)
        if not isinstance(selected, dict):
            raise ValueError("--select must be a JSON object")
        selected = {str(k): str(v) for k, v in selected.items()}
        with args.csv.open(newline="", encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f) if matches(row, selected)]
        if not rows:
            raise ValueError("selection has no rows")
        if args.kind == "ranks":
            groups = {tuple(row[k] for k in GROUP) for row in rows}
            if len(groups) != 1:
                raise ValueError(f"selection has {len(groups)} configurations; narrow --select including step")
            if any(row["correct"] != "PASS" for row in rows):
                raise ValueError("cannot chart failed correctness")
            walls = [r for r in rows if r["record"] == "wall"]
            chosen = min(walls, key=lambda r: positive(r["elapsed_s"]))
            ranks = sorted((r for r in rows if r["record"] == "rank" and
                            r["repetition"] == chosen["repetition"]), key=lambda r: int(r["rank"]))
            if len(ranks) != int(chosen["processes"]):
                raise ValueError("incomplete rank records")
            subtitle = (f"run={chosen['run_id']} rep={chosen['repetition']} size={chosen['size']} "
                        f"p={chosen['processes']} t={chosen['threads']} step={chosen['step']}; "
                        f"slowest-rank elapsed={chosen['elapsed_s']} s")
            svg_bars(f"{chosen['question']} {chosen['version']}: per-rank times", subtitle,
                     [r["rank"] for r in ranks],
                     [(label, [metric(r[field]) for r in ranks]) for label, field in
                      [("Compute", "compute_s"), ("MPI calls (includes wait)", "comm_s"), ("Exposed wait", "wait_s")]],
                     "seconds; grouped bars, not additive; NA = unavailable", args.output)
        else:
            comparison = {(r["baseline_run_id"], r["baseline_version"], r["baseline_experiment"],
                           r["question"], r["mode"], r["size"], r["steps"], r["step"], r["vector_bytes"])
                          for r in rows}
            if len(comparison) != 1:
                raise ValueError("scaling chart must use one workload and baseline")
            rows.sort(key=lambda r: (int(r["processes"]) * int(r["threads"]), r["version"]))
            labels = [f"{r['version']} {r['processes']}x{r['threads']}" for r in rows]
            row = rows[0]
            svg_bars(f"{row['question']} scaling, size={row['size']}",
                     f"baseline {row['baseline_version']}, run={row['baseline_run_id']}; step={row['step']}",
                     labels, [("Speedup", [positive(r["speedup"]) for r in rows]),
                              ("Efficiency", [positive(r["efficiency"]) for r in rows])],
                     "ratio; efficiency denominator = MPI ranks x OpenMP threads", args.output)
        print(f"wrote {args.output}")
    except (ValueError, KeyError, OSError, csv.Error) as exc:
        parser.exit(1, f"plot failed: {exc}\n")


if __name__ == "__main__":
    main()
