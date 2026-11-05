# Requirement and evidence map

Original assignment formulas/tables are the reference specification. All
computational kernels are C, MPI and OpenMP; Python is orchestration/analysis
only. There are 13 executables and **13 rubric rows** including cluster/report.

| Rubric item | Marks | Implementation and verification | Real cluster evidence needed |
|---|---:|---|---|
| Cluster | 5 | runner evidence collects actual host mapping, CPU, RAM, affinity, compiler/MPI | two or more actual nodes; VM configuration and launch evidence |
| 1.1 | 6 | Q1 V1/V2/V3; independent complete output checks | `q1-loop-order`, `q1-openmp`; N1000/2000 and physical-core thread sweep |
| 1.2 | 8 | Q1 V4; stage0 Scatterv,1 Bcast,2 multiply,3 Gatherv; memory.py | `q1-mpi-comparison`, A at N2000/4000,p1/2/4/8; analytical N32768 memory |
| 1.3 | 8 | Q1 V5; even/odd blocking send/receive, variable blocks | compare A/B times and memory; explain blocking-send deadlock |
| 1.4 | 8 | Q1 V6; two B buffers, Irecv/Isend, read-only current block, Waitall | C wait vs B MPI-call time; `q1-hybrid` and one rank/core N4000 |
| 2.1 | 6 | Q2 V1; exact CSR generator, duplicates/self-links, T10 | `q2-openmp`, static/dynamic64; printed links/thread; n1million |
| 2.2 | 7 | Q2 V2; local CSR, Allgatherv every step, exact payload bytes | `q2-allgatherv`, p1/2/4/8; per-step max-local intervals and total/10 |
| 2.3 | 10 | Q2 V3; 128-row periodic halos; p1/p2 cases; overlap/wait-first | compare `q2-halo-overlap` vs `q2-halo-wait-first` and allgather |
| 2.4 | 7 | Q2 V4; root running-degree partition, Bcast, Gather metrics | equal rows vs equal links p4/8; `q2-hybrid` vs V1 matching workload |
| 3.1 | 7 | Q3 V1/V2; local examples, complete gradient/weight oracle | `q3-correctness` m512,p1/2/3/4; m1024 per-step computation/reduction |
| 3.2 | 11 | Q3 V3; exact signature ring_allreduce; uneven/empty chunks; Allreduce self-test | `q3-micro`: 1024/1048576/67108864 bytes,p2/4/8, all3 methods |
| 3.3 | 7 | Q3 all3 candidates measured with batch512 and m1024 | select fastest actual candidate; `q3-scaling-candidates` + `q3-hybrid-candidates`; own p1 baseline |
| Report | 10 | report outline and deterministic analysis/plots tools | <=10-page PDF with actual tables, plots, discussions, printed checks |

The assignment rubric totals100 marks: cluster +4 Q1 +4 Q2 +3 Q3 +report =13
rows. The individual rows above follow the original rubric table.

## Explicit correctness coverage

`./run.sh smoke` contains34 commands: all13 executables, N8 full matrix,
Q1 zero-row ranks, Q2 schedules/halos/balancing, Q3 complete tiny training,
nondivisible vectors and vector lengths less than p, including p8.
`./run.sh reference` checks the official Q1/Q2/Q3 numeric tables; the full
matrix additionally exercises Q1's larger N2000/4000 values. Each executable
also gates and validates each measured repetition. Correctness failure is a
nonzero exit, never a fast result labelled passing.

Local development checked p1/2/3/4/8; required full benchmark processes are
p1/2/4/8. A default benchmark alone does not replace the uneven p3 correctness
checks. Follow the smoke -> reference -> full workflow in README.

## Report consistency checks

- Compare identical dimensions, batch size, iteration count, and units.
- Use complete best-of-five **whole-run wall** minima for speedup; divide Q2
  by10/Q3 by5 only when reporting per-step average of the whole run.
- For rank bars choose the **same winning repetition** as the wall time;
  do not independently minimize every rank's compute/communication samples.
- Explicitly choose the sequential baseline (Q1 serial-ikj, Q2 V1 t1,
  Q3 selected method p1,t1), and record its run_id.
- State efficiency denominator p*threads and do not substitute node count.
- Check observed nodes, actual threads, affinity, RAM pressure, stderr,
  missing commands and correctness logs before drawing conclusions.
- Discuss algorithm choices using actual evidence. No script concludes that
  one strategy wins on a cluster it has never run on.
