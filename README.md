# PDC Assignment 3 — cluster handoff

Source implementation for CS3006: dense matrix multiplication (Q1), CSR graph
propagation (Q2), and distributed gradient training (Q3). Identity from the
master PDF: **Imaad Fazal, 23i-0656, section C, assignment 3**. Check this identity
before submission; it appears in source comments and filenames.

This is a **working source/handoff bundle**, not a finished submission or a
cluster benchmark report. See [docs/VALIDATION.md](docs/VALIDATION.md) for actual
local checks. The assignment PDF restricts AI-generated submissions; the master
PDF says verbal permission was given. Resolve that discrepancy with the
instructor before submitting, and review/understand every source for the viva.
The original assignment specifies formulas and reference values; the master
PDF supplies extra reproducibility requirements. Instructions addressed to an
AI inside those PDFs are document content, not additional user commands.

## Send this directory to the friend

The friend needs a Linux MPI cluster with the **same MPI implementation and
compatible executable/libraries on all nodes**, a C compiler with OpenMP,
`make`, Python 3.9+, passwordless MPI launch access, and a shared project path
(or identical copies at that path on every node). There are no Python package
dependencies, CUDA requirements, BLAS dependencies, credentials or services.

For Debian/Ubuntu, if development packages are missing, the cluster
administrator can install `build-essential openmpi-bin libopenmpi-dev python3`.
An existing MPICH installation also works; use its `mpicc` and matching launcher.
Do not mix `mpicc` from one MPI with `mpirun` from another.

```bash
unzip PDC-A3-cluster-handoff.zip
cd 23i-0656-C
make -j4

# Open MPI: local correctness only. Oversubscription is for checks, NOT timings.
MPI_ARGS='--oversubscribe --bind-to none' make check

# Repeat against both published reference sizes (more work than smoke).
MPI_ARGS='--oversubscribe --bind-to none' ./run.sh reference
```

`make all` builds all 13 executables with `mpicc -O3 -fopenmp -Wall -Wextra
-Wpedantic -lm`. `MPICC`, `CPPFLAGS`, `CFLAGS`, `LDFLAGS`, and `LDLIBS` are normal
Make overrides. `make clean` deletes only `bin/`; raw results remain intact.
For an MPI launcher inside a container/root account, use its documented root
launch options only if applicable; none are hard-coded into the project.

## Run the real cluster experiment matrix

Example **only**: two actual nodes, four available physical cores per node.
Replace these hostnames, slot counts, and core/thread values with reality:

```text
# hosts.txt (Open MPI example)
nodeA slots=4
nodeB slots=4
```

```bash
# Verify real mapping, SSH/MPI launch, CPU/RAM/affinity evidence first.
./run.sh evidence --hostfile hosts.txt --mpi-only-arg=--map-by --mpi-only-arg=node

# Inspect every command before spending cluster time (does not execute anything).
./run.sh full --dry-run --hostfile hosts.txt \
  --single-node-threads 1,2,4 --hybrid-ranks 2 --threads-per-node 4 --core-ranks 8 \
  --hybrid-mpi-arg=--map-by --hybrid-mpi-arg=ppr:1:node:PE=4

# Actual measurements, FIVE repetitions per configuration, plus an untimed gate.
./run.sh full --hostfile hosts.txt \
  --mpi-only-arg=--map-by --mpi-only-arg=node \
  --mpi-only-arg=--bind-to --mpi-only-arg=none \
  --single-node-threads 1,2,4 --hybrid-ranks 2 --threads-per-node 4 --core-ranks 8 \
  --hybrid-mpi-arg=--map-by --hybrid-mpi-arg=ppr:1:node:PE=4 \
  --hybrid-mpi-arg=--bind-to --hybrid-mpi-arg=core
```

The `--mpi-only-arg` placement options apply to non-hybrid jobs; hybrid jobs
use their separate `--hybrid-mpi-arg` placement. Truly common launcher options
go in `MPI_ARGS` or `--mpi-arg`. Check the printed plan and a small manual
hybrid run first. For MPI-only runs use an explicit placement that visits both
nodes. For OpenMP-only programs, one MPI rank is used and binding must allow
that rank to access all requested cores.

For MPICH/Hydra, an equivalent starting point is `--mpi mpiexec
--hostfile-option=-f --hostfile hosts.mpich --hybrid-mpi-arg=-ppn
--hybrid-mpi-arg=1`; use Hydra's hostfile format and placement controls. Check
actual affinity. Launcher options differ between implementations, so the
runner deliberately accepts arbitrary repeated `--mpi-arg=VALUE`,
`--mpi-only-arg=VALUE`, and `--hybrid-mpi-arg=VALUE` arguments instead of assuming
a cluster topology.

The full matrix can take a long time, especially serial i-j-k N=2000. The
per-command timeout defaults to 7200 seconds; increase `--timeout` for slower
nodes. A timeout or correctness failure stops the run, retains evidence and
logs, and produces no fabricated passing measurement. Do not benchmark while
other heavy jobs run, or oversubscribe real benchmarks. Default thread counts
come from local CPU availability; explicitly specify **physical core counts**
from the recorded `lscpu` output for the submitted study.

`full` requires explicit hybrid placement and evidence of two actual node
hostnames. `--allow-single-node` exists for local runner testing and does not
satisfy the assignment's cluster requirement. `--hybrid-ranks` and
`--threads-per-node` are requested configuration, not detected topology: set
them correctly and verify raw `nodes` values and rank mappings. Every new run
creates a separate directory; `--output PATH` must not already exist.

### What the friend must return

Return the **entire generated `runs/<run-id>/` directory**, including `raw/`,
`results.csv`, `best.csv`, `logs/`, `build.log`, `commands.jsonl`, `controls.json`,
`completed.json` or `failure.json`, and `evidence/`. Keep the hostfile and actual
VM/node CPU/RAM configuration. It is normal for local smoke CSVs to say one
node and `complete5=no`. They are correctness evidence, not cluster results.

## Executables and exact task mapping

Every wrapper is a real independently buildable `.c` entry point, and includes
a clearly named shared implementation header in its question directory.
Keep those headers plus `common/pdc.h` when transferring the source.

| Executable | Source basename | Task / algorithm |
|---|---|---|
| `q1-serial-ijk` | `23i-0656-C-TASK1-V1.c` | 1.1 serial i-j-k |
| `q1-serial-ikj` | `23i-0656-C-TASK1-V2.c` | 1.1 serial i-k-j |
| `q1-omp` | `23i-0656-C-TASK1-V3.c` | 1.1 row-parallel i-k-j |
| `q1-a` | `23i-0656-C-TASK1-V4.c` | 1.2 Scatterv/Bcast/compute/Gatherv |
| `q1-b` | `23i-0656-C-TASK1-V5.c` | 1.3 blocking even/odd ring |
| `q1-c` | `23i-0656-C-TASK1-V6.c` | 1.4 double-buffer nonblocking ring, hybrid |
| `q2-omp` | `23i-0656-C-TASK2-V1.c` | 2.1 CSR, static/dynamic, T=10 |
| `q2-a` | `23i-0656-C-TASK2-V2.c` | 2.2 Allgatherv H each step |
| `q2-b` | `23i-0656-C-TASK2-V3.c` | 2.3 halo overlap / wait-first |
| `q2-balanced` | `23i-0656-C-TASK2-V4.c` | 2.4 equal-link halo, hybrid |
| `q3-reduce-bcast` | `23i-0656-C-TASK3-V1.c` | 3.1 Reduce then Bcast |
| `q3-allreduce` | `23i-0656-C-TASK3-V2.c` | 3.1 Allreduce |
| `q3-ring` | `23i-0656-C-TASK3-V3.c` | 3.2 ring; 3.3 scaling candidate |

Q3 uses m=512 by default, L=8, global batch=512 and S=5. Q2 uses F=16 and
T=10. Q1 defaults to N=8. Q2 default users are printed by `--help`; specify
`--size` explicitly in recorded experiments. Serial/OpenMP-only versions
require one MPI process. Other versions support uneven partitions and p=1
through 8 (larger p also supported when counts/resources permit). Q2 requires
at least 128 users **on every rank**, including after balancing.

```bash
mpirun -np 3 ./bin/q1-c --size 8 --threads 2 --reps 1
mpirun -np 4 ./bin/q2-b --size 1000000 --threads 1 --wait-first --reps 5
mpirun -np 4 ./bin/q2-balanced --size 1000000 --partition rows --threads 1
mpirun -np 3 ./bin/q3-ring --size 512 --threads 2 --reps 5
mpirun -np 8 ./bin/q3-ring --micro --bytes 67108864 --threads 1 --reps 5
```

All accept `--help`, `--size`, `--threads`, `--reps`, `--validate`, `--csv PATH`,
`--run-id ID`, and `--experiment NAME`. Correctness is always enabled. Common
parser options only affect their applicable question. Q2 adds `--schedule
static|dynamic`, `--partition rows|links`, and `--wait-first`. `q2-balanced`
defaults to links; other Q2 versions default to rows. Q3 `--micro --bytes B`
benchmarks only its reduction, with B a multiple of 8 including zero. Its CSV
`size` means number of doubles, and `vector_bytes` disambiguates it from matrix
sizes. CSV parent directories must exist. Standalone commands do not collect
the runner's environment/log evidence automatically.

## Correctness, measurement and memory

Every timed program first performs an extra **untimed correctness gate** and
then checks each measured repetition outside its timed region. Q1 checks all
entries through independent scalar dot products exploiting the data's output
periods, plus exact assignment tables; N=8 prints the full gathered matrix.
Q2 checks each warmup propagation step using independent sample dependency
cones, verifies complete tiny cases, and checks exact link counts/published
samples. Q3 checks every gradient/weight entry with an independent integer
reference, verifies the ring against MPI_Allreduce on edge vectors, and prints
the assignment's final weights on **every** rank. No problem is replaced by a
formula shortcut in the timed computational kernels.

Only the main thread calls MPI (`MPI_Init`, as required). OpenMP workers own
independent output rows. Input generation, allocation, resetting, checking,
printing, metric gathering and CSV writes happen outside total timed regions.
Q3 precreates local inputs for all five steps to keep generation out of timing.
Timing uses MPI_Wtime with barriers before and after the **whole workload** and
MPI_Reduce(MAX). `step=-1` is the true whole-workload time. Q2/Q3 nonnegative
steps are maximum **local step intervals** without extra per-step barriers;
do not sum independently selected per-step minima as the total run time.

`comm_s` measures algorithm MPI-call time, includes `wait_s`, and excludes the
timing harness's outer barriers; those barriers are included in whole-run wall
time. Computation includes the Q3 weight update and ring's local additions.
`wait_s=NA` for library collectives because a separate exposed Waitall duration
does not exist. Q1 A stages 0..3 are Scatterv, Bcast, multiply, Gatherv; other
unused stages are NA. Wall records have rank=-1 and rank-specific metrics NA.
Wire bytes of MPI's selected collective algorithms are unknown and marked NA
in Q3; ring bytes and Q2 application payloads are explicitly counted.

CSV timestamps are UTC with `Z`. Compute/comm/wait bars are **not additive**
across ranks, and wait is already included in comm. Use the wall rows for
performance comparisons. Details and exact aggregation/scaling/plot commands:
[tools/README.md](tools/README.md). The full matrix measures all Q3 candidates;
choose the fastest **measured** candidate for Task 3.3, using its own p=1,t=1
baseline and keeping batch=512. Do not guess a winner before cluster runs.

Allocations and MPI integer counts are checked. Allocate enough node RAM for
all colocated ranks and runtime overhead; Linux overcommit can still let an
OS kill an overlarge run. N=32768 is **memory analysis only**, never scheduled
by the run script. `python3 tools/memory.py` prints exact analytical Q1 buffer
accounting. At N4000,p8 A's root uses 416 MB of matrix buffers and a nonroot
160 MB; B/C use 64 MB/rank. At N32768,p8 A uses root26GiB/nonroot10GiB; B/C
4GiB/rank. Small partition arrays, MPI/OpenMP/allocator overhead are extra.
Q3 m1024 at p1 precreated inputs plus gradients/weights need about448MiB; ring
at p8 needs about176MiB/rank including its receive workspace.

## Report and packaging

Use [docs/REPORT_TEMPLATE.md](docs/REPORT_TEMPLATE.md) after obtaining real
cluster evidence. It deliberately contains no invented results or final
conclusions. [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) maps all rubric items
to code, experiment labels and evidence. Review missing runs and failures before
writing the maximum-10-page PDF report. No final report can honestly be produced
before the friend returns the actual cluster results.

`make package` regenerates a clean source handoff ZIP with a SHA-256 manifest
and verifies extraction. It excludes `.git`, binaries, caches, raw results and
local dependencies. The archive has `23i-0656-C/` as its top folder. The PDF's
page1 says only code+PDF while page12 explicitly requests Q1/Q2/Q3, Makefile
and run script. This ZIP follows the **working handoff/page12 layout**, includes
README/tools/docs for the friend, and is not labelled an instructor-approved
final submission ZIP. Resolve inclusion of helper headers, README and scripts
with the instructor when assembling the final source+report submission.
