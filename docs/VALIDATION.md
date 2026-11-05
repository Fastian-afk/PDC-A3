# Validation performed before handoff

These are **single-machine correctness/build checks**, not Beowulf performance
measurements or evidence of two-node execution. Hardware-specific times are
deliberately not presented as assignment benchmark results.

Environment used: Debian13, GCC14.2, Open MPI5.0.7, OpenMP. This development
image had the MPI runtime without mpi.h, so its matching Debian-signed
libopenmpi-dev5.0.7-1 package was extracted into a local dependency directory.
Compilation used temporary CPPFLAGS/LDFLAGS overrides. No local dependency
paths, headers or libraries are shipped or required in the source ZIP: the
friend should use the cluster's normal matching MPI development installation.

| Check | Observed result |
|---|---|
| All13 C executables, optimized OpenMP + warning flags | compiled without warnings |
| Q1 six algorithms; p1/2/3/4/8; sizes1/5/8/17/65 | passed full output oracles, including zero-row ranks and uneven ring blocks |
| Q1 hybrid/repeated runs and N1000 A/B/C | passed Table1(b) and complete output checks |
| Q1 largest required N4000, C,p3 | passed all outputs and samples -7,5,20,9,17 |
| Q1 actual-thread metadata, requested2 with thread-limit1 | correctly recorded1 thread; normal2-thread check also passed |
| Q2 four algorithms, schedules, p1/2/3/4/8, overlap/wait-first, uneven/balanced rows | passed; invalid <128-users/rank request rejected |
| Q2 n100000 and n1000000, all four variants | exact link counts1020227/10160271 and all published Table3 samples matched |
| Q2 minimum128/rank and explicit partition override | passed |
| Q2 final thread-limit1, requested2,p3, two repetitions | passed, correctly recorded1 thread |
| Q3 three training methods, tiny m9,p1/2/3/4/8 | complete gradient/weight reference checks passed |
| Q3 ring vectors0/1/3/13/128 doubles, p1/2/3/4/8 | passed independent MPI_Allreduce comparisons and exact expected sums |
| Q3 m512, all methods,p1/2/3/4 | Table4 plus full gradient/weight checks passed |
| Q3 m1024, all methods,p3 | Table4 plus full gradient/weight checks passed |
| Q3 64MiB microbenchmark, all methods,p3 and8 | complete reduced vector validation passed |
| Q3 actual-thread metadata, capped team | correctly recorded actual team size |
| Integrated final smoke workflow | all34 commands passed;460 raw rows,154 measurement groups; aggregation completed |
| Results tools |11 regression tests passed, including NA wait values and zero-byte vectors |
| Actual CSV analysis | aggregation, explicit-baseline scaling and SVG creation passed |
| Deterministic regeneration | regenerated merged/best CSV files byte-identical |
| Full plan for example2nodes x4cores |114 commands with five repetitions, separate hybrid placement and N4000 sequential baseline |
| Handoff package | ZIP CRC and extracted source SHA-256 verification performed by tools/package.py |

The full multi-node experiment plan, SSH/MPI connectivity on the friend's
cluster, network overlap benefits, best-of-five cluster timings, speedups,
efficiencies, winning algorithms, final plots and the final <=10-page PDF
report have **not** been measured/completed here. They require the actual
cluster and returned run directory. Local oversubscription was used only to
check p8 correctness. No performance conclusions are inferred from it.

Reproduce the portable checks with `make` then `MPI_ARGS='--oversubscribe
--bind-to none' make check` on Open MPI, and
`python3 tools/test_results_tools.py`. Use appropriate launcher options for
other MPI implementations. The reference mode exercises the published numeric
tables; full mode provides the actual performance study for the friend to run.
