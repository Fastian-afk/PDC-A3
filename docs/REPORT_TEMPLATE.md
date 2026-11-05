# Working report outline — fill only after real cluster runs

This is an outline, **not** a completed report or empirical results. Produce a
PDF of at most10 pages with your own verified measurements and explanations.

1. **Identity and cluster (~1 page).** Name/ID/section; actual nodes, physical
   and logical cores, RAM, VM allocation, interconnect, compiler/MPI versions,
   affinity, hostfile and evidence of execution on at least two real nodes.
2. **Q1 (~3 pages).** Tables of printed reference outputs and best-of-five
   timings; loop-order/OpenMP comparison; A's four stages; A/B/C comparison;
   root vs nonroot analytical memory N4000 and32768,p8; deadlock, ownership,
   buffer lifetime, overlap and rank/node versus rank/core discussion.
3. **Q2 (~3 pages).** Correct graph link counts and three checked outputs;
   scheduling/link counts per thread; allgather versus halo bytes/time; wait
   first versus overlap; equal-row/equal-link rank bars; hybrid speedup with
   matching sequential baseline.
4. **Q3 (~2 pages).** Step0 gradient and final weight checks; per-step
   computation/reduction for library collectives; ring versus Allreduce and
   Reduce+Bcast across all vector sizes/processes; fixed-batch scaling and
   hybrid results for the measured best method.
5. **Reproducibility and conclusions (~1 page).** Actual winning strategies,
   measured limits, run IDs, timing protocol, best-of-five selection and
   exact repeat commands. Explain all observed failures/omitted trials.

Suggested table columns (fill from evidence, never invented):

| Question/task | Version | Size | Actual nodes | p x threads | Checked values | Best total seconds | Seconds/step | Speedup | Efficiency | Winning run_id/repetition |
|---|---|---|---|---|---|---|---|---|---|---|
| Fill after cluster runs | | | | | | | | | | |

For rank bar charts use tools/plots.py. Keep computation, MPI calls and exposed
wait grouped rather than stacked. State that wait is included in MPI-call time
and timing-barrier overhead is included only in wall time. Include elapsed
speedup/efficiency plots generated from tools/scaling.py with explicit baselines.

Before exporting PDF: count pages, verify identities/source filenames, reconcile
every task in REQUIREMENTS.md, inspect numerical output side by side, retain raw
CSV/logs, and check the final ZIP after extraction. Cluster results and the PDF
remain outstanding until the friend's real run evidence is available.
