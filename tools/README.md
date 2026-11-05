# Results and report tools

These Python 3 tools use only the standard library. Run them from the project
root. The examples assume the cluster run directory is `runs/my-run`; replace
that path and the JSON selectors with the configuration you actually ran.

## Aggregate the raw measurements

```sh
python3 tools/aggregate.py runs/my-run/raw \
  --output runs/my-run/results.csv \
  --best runs/my-run/best.csv --expected-reps 5 --require-complete
```

For a one-repetition smoke run, use `--expected-reps 1`. Its `complete5` field
remains `no`; a smoke check is not a best-of-five benchmark.

`results.csv` contains all original field values plus `source_file` and
`source_line`. `best.csv` selects the minimum **wall** record per configuration
and step, retaining the selected repetition and provenance. `step=-1` is a whole
workload; nonnegative steps are individual propagation/training steps. The
whole-workload `elapsed_per_step_s` divides total elapsed by `steps`; it does not
sum independently selected per-step minima.

Aggregation rejects failed correctness, malformed or inconsistent schemas,
negative/nonfinite timings, incomplete rank sets, wall times that are not the
maximum rank duration, and duplicate measurement identities. Use a new `run_id`
for a new independent run. `--deduplicate-identical` is an explicit option for
byte-identical copied records; conflicting duplicates always fail. The check for
five repetitions covers configurations present in the input; compare the run
plan/logs as well to identify experiments that never produced any CSV rows.

## Calculate speedup and efficiency

First inspect `best.csv` to find the baseline's actual `run_id` and experiment.
Use one process and one thread for the baseline. For example:

```sh
python3 tools/scaling.py runs/my-run/best.csv \
  --baseline '{"run_id":"ACTUAL_BASELINE_RUN_ID","question":"Q1","version":"serial-ikj","size":"4000","step":"-1"}' \
  --targets '{"question":"Q1","size":"4000","step":"-1"}' \
  --output runs/my-run/q1-scaling.csv
```

The baseline selector must resolve to exactly one row. Add `experiment` or other
fields if necessary. Every target must have the same question, mode, size,
number of steps, selected step and vector byte count as the baseline. The
default requires five repetitions in both baseline and targets.
`--allow-incomplete` exists for testing the analysis on smoke measurements.

Speedup is `baseline_elapsed_s / best_elapsed_s`. Efficiency is
`speedup / (MPI_processes * OpenMP_threads)`, so its denominator is explicit.
The output carries baseline and target source-file provenance.

## Generate charts from measured data

```sh
python3 tools/plots.py runs/my-run/results.csv --kind ranks \
  --select '{"question":"Q1","version":"C","size":"4000","processes":"8","threads":"1","step":"-1"}' \
  --output runs/my-run/q1-rank-times.svg

python3 tools/plots.py runs/my-run/q1-scaling.csv --kind scaling \
  --output runs/my-run/q1-scaling.svg
```

Narrow `--select` until it identifies one configuration; add `run_id` and
`experiment` if needed. The rank chart chooses the minimum wall-time repetition
of that configuration and draws its compute, MPI-call, and exposed-wait bars.
MPI-call time **includes** wait time. Bars are grouped, not stacked; overlapping
durations must not be added together. Scaling charts require one workload and
one baseline. SVGs include values as tooltip text and can be imported into a
report or converted to PDF by a standard browser/vector editor.
An unavailable metric, such as separately exposed wait inside a blocking
collective, stays `NA` in CSVs and is labelled `NA` in charts, never zero.

## Analytical Q1 memory accounting

```sh
python3 tools/memory.py --size 4000 32768 --processes 8 > q1-memory.csv
```

This is allocation arithmetic, clearly marked `analytical`, not measured RSS.
Let `r` be the rank's rows and `rmax=ceil(N/p)`. Version A's matrix buffers need
`8*(N*N + 2*r*N)` bytes on every rank, with an additional `16*N*N` bytes on root
for full A and gathered C. Versions B and C each need
`8*(2*r*N + 2*rmax*N)` bytes; both reuse two B buffers. The tool also includes
two `int[p]` partition arrays (using the native C int ABI), and the temporary root
checking matrix for N=8. MPI internals, runtime/allocator overhead, OpenMP stacks
and small post-timing metric gathers are excluded. Node capacity must account
for **all** ranks placed on that node. Zero-row ranks retain the allocator's
one-byte dummy allocations for each empty local matrix.

## Regression tests

```sh
python3 tools/test_results_tools.py
```

The tests use explicitly synthetic CSV fixtures in temporary directories. They
exercise rejection of corrupt, failed, duplicated or incomplete evidence,
whole-run versus per-step separation, deterministic regeneration and chart
generation. Fixture values are not measurements and are never report evidence.
