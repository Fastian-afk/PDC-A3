#!/usr/bin/env python3
"""Imaad Fazal | 23i-0656 | Section C | Assignment 3.
Execute auditable smoke/reference/full experiment plans, with no shell eval.
"""
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def numbers(text):
    try:
        values = [int(x) for x in text.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('use comma-separated positive integers') from exc
    if not values or any(x < 1 for x in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError('values must be positive and unique')
    return values


def default_threads():
    cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    values, t = [], 1
    while t <= cores:
        values.append(t)
        t *= 2
    if values[-1] != cores:
        values.append(cores)
    return values


def plan(args):
    jobs = []

    def add(executable, size, p=1, threads=1, experiment='', extra=(), hybrid=False):
        jobs.append(dict(executable=executable, size=size, processes=p,
                         threads=threads, experiment=experiment,
                         extra=list(extra), hybrid=hybrid))

    methods = ['reduce-bcast', 'allreduce', 'ring']
    if args.mode == 'smoke':
        for version in ['serial-ijk', 'serial-ikj', 'omp', 'a', 'b', 'c']:
            add('q1-' + version, 8, threads=2 if version == 'omp' else 1, experiment='q1-table1a')
        for p in [2, 3]:
            for v in ['a', 'b', 'c']:
                add('q1-' + v, 8, p, experiment='q1-table1a')
        for v in ['a', 'b', 'c']:
            add('q1-' + v, 1, 3, experiment='q1-zero-row')
        for schedule in ['static', 'dynamic']:
            add('q2-omp', 2048, threads=2, experiment='q2-schedule', extra=['--schedule', schedule])
        for v in ['a', 'b', 'balanced']:
            for p in [1, 3]:
                add('q2-' + v, 2048, p, 2, 'q2-small')
        add('q2-b', 1024, 3, 2, 'q2-wait-first', ['--wait-first'])
        for v in methods:
            for p in [1, 3]:
                add('q3-' + v, 9, p, 2, 'q3-small')
        for p, b in [(3, 0), (3, 8), (3, 104), (8, 24)]:
            add('q3-ring', 9, p, 1, 'q3-ring-edge', ['--micro', '--bytes', str(b)])
        return jobs

    if args.mode == 'reference':
        for v in ['serial-ijk', 'serial-ikj', 'omp', 'a', 'b', 'c']:
            add('q1-' + v, 8, experiment='q1-table1a')
        for v in ['a', 'b', 'c']:
            add('q1-' + v, 1000, 3, 2, 'q1-table1b')
        for n in [100000, 1000000]:
            for v in ['omp', 'a', 'b', 'balanced']:
                add('q2-' + v, n, 1 if v == 'omp' else 3, 2, 'q2-table3')
        for m in [512, 1024]:
            for v in methods:
                for p in ([1, 2, 3, 4] if m == 512 else [3]):
                    add('q3-' + v, m, p, 2, 'q3-table4')
        return jobs

    # All mandatory problem sizes; five repetitions are the default for full.
    for n in [1000, 2000]:
        for v in ['serial-ijk', 'serial-ikj']:
            add('q1-' + v, n, experiment='q1-loop-order')
        for t in args.single_node_threads:
            add('q1-omp', n, threads=t, experiment='q1-openmp')
    # Table 2 / cross-version comparison also needs the identical N=4000
    # sequential baseline, rather than reusing a smaller problem's time.
    add('q1-serial-ikj', 4000, experiment='q1-large-baseline')
    for t in args.single_node_threads:
        add('q1-omp', 4000, threads=t, experiment='q1-large-openmp')
    for n in [2000, 4000]:
        for p in args.ranks:
            for v in ['a', 'b', 'c']:
                add('q1-' + v, n, p, 1, 'q1-mpi-comparison')
    add('q1-c', 4000, args.hybrid_ranks, args.threads_per_node, 'q1-hybrid', hybrid=True)
    if args.core_ranks not in args.ranks:
        add('q1-c', 4000, args.core_ranks, 1, 'q1-rank-per-core')
    for t in args.single_node_threads:
        for schedule in ['static', 'dynamic']:
            add('q2-omp', 1000000, threads=t, experiment='q2-openmp', extra=['--schedule', schedule])
    for p in args.ranks:
        add('q2-a', 1000000, p, 1, 'q2-allgatherv')
        add('q2-b', 1000000, p, 1, 'q2-halo-overlap')
        add('q2-b', 1000000, p, 1, 'q2-halo-wait-first', ['--wait-first'])
    for p in [4, 8]:
        add('q2-balanced', 1000000, p, 1, 'q2-equal-links')
    add('q2-balanced', 1000000, args.hybrid_ranks, args.threads_per_node, 'q2-hybrid', hybrid=True)
    for v in methods:
        for p in [1, 2, 3, 4]:
            add('q3-' + v, 512, p, 1, 'q3-correctness')
        for p in args.ranks:
            add('q3-' + v, 1024, p, 1, 'q3-scaling-candidates')
        # Run all candidates; select the measured winner in the analysis tool.
        add('q3-' + v, 1024, args.hybrid_ranks, args.threads_per_node,
            'q3-hybrid-candidates', hybrid=True)
        for p in [2, 4, 8]:
            for b in [1024, 1048576, 67108864]:
                add('q3-' + v, 1024, p, 1, 'q3-micro', ['--micro', '--bytes', str(b)])
    return jobs


def launch_prefix(args, job=None):
    prefix = [args.mpi] + shlex.split(os.environ.get('MPI_ARGS', '')) + args.mpi_arg
    if args.hostfile:
        prefix += [args.hostfile_option, str(Path(args.hostfile).resolve())]
    if job is not None:
        prefix += args.hybrid_mpi_arg if job['hybrid'] else args.mpi_only_arg
        prefix += ['-np', str(job['processes'])]
    return prefix


def command_for(args, job, csv, run_id):
    return launch_prefix(args, job) + [str(ROOT / 'bin' / job['executable']),
        '--size', str(job['size']), '--threads', str(job['threads']),
        '--reps', str(args.reps), '--csv', str(csv), '--run-id', run_id,
        '--experiment', job['experiment']] + job['extra']


def capture(command, path, timeout=60, env=None):
    with path.open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                cwd=ROOT, env=env, start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return 124


def evidence(args, out):
    data = dict(timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
                local_hostname=socket.gethostname(), local_logical_cpus=os.cpu_count(),
                local_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                mpi=args.mpi, mpi_args=args.mpi_arg, mpi_only_args=args.mpi_only_arg,
                hybrid_mpi_args=args.hybrid_mpi_arg,
                hostfile=args.hostfile, note='Availability is not proof of a multi-node run.')
    for name, cmd in [('mpi', [args.mpi, '--version']), ('compiler', ['mpicc', '--version']),
                      ('local_cpu', ['lscpu']), ('local_ram', ['free', '-h'])]:
        data[name + '_exit'] = capture(cmd, out / (name + '.txt')) if shutil.which(cmd[0]) else 'unavailable'
    if args.hostfile:
        shutil.copyfile(args.hostfile, out / 'hostfile.used')
    probe_p = max(args.ranks + [args.hybrid_ranks, args.core_ranks])
    probe = dict(processes=probe_p, hybrid=False)
    data['mapping_exit'] = capture(launch_prefix(args, probe) + ['hostname'], out / 'rank_hosts.txt')
    hosts = (out / 'rank_hosts.txt').read_text().splitlines() if not data['mapping_exit'] else []
    # MPI warnings make this conservative rather than inflating the node count.
    valid_hosts = [h for h in hosts if h and all(c.isalnum() or c in '-._' for c in h)]
    data['mapped_unique_hosts'] = sorted(set(valid_hosts)) if len(valid_hosts) == probe_p else []
    data['mapped_nodes'] = len(data['mapped_unique_hosts']) if len(valid_hosts) == probe_p else None
    # Record per-rank CPU/memory/affinity on actual execution hosts; no SSH assumptions.
    info = 'echo "HOST=$(hostname) PID=$$"; getconf _NPROCESSORS_ONLN; free -b; taskset -pc $$; lscpu'
    data['remote_info_exit'] = capture(launch_prefix(args, probe) + ['sh', '-c', info], out / 'nodes.txt')
    (out / 'environment.json').write_text(json.dumps(data, indent=2) + '\n')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['smoke', 'reference', 'full', 'evidence'])
    parser.add_argument('--mpi', default=os.environ.get('MPIEXEC', 'mpirun'))
    parser.add_argument('--mpi-arg', action='append', default=[], help='repeat; use --mpi-arg=--oversubscribe')
    parser.add_argument('--mpi-only-arg', action='append', default=[], help='placement for non-hybrid jobs only')
    parser.add_argument('--hybrid-mpi-arg', action='append', default=[], help='launcher-specific one-rank/node placement')
    parser.add_argument('--hostfile')
    parser.add_argument('--hostfile-option', default='--hostfile', help='MPICH users may choose -f')
    parser.add_argument('--ranks', type=numbers, default=[1, 2, 4, 8])
    parser.add_argument('--single-node-threads', type=numbers, default=default_threads())
    parser.add_argument('--hybrid-ranks', type=int, default=2)
    parser.add_argument('--threads-per-node', type=int, default=4)
    parser.add_argument('--core-ranks', type=int, help='actual total cores used in rank/core comparison')
    parser.add_argument('--reps', type=int)
    parser.add_argument('--timeout', type=float, default=7200, help='seconds PER command, including gate/repetitions')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='print commands; no build, MPI, measurements, or files')
    parser.add_argument('--allow-single-node', action='store_true', help='local validation only; does not meet cluster rubric')
    args = parser.parse_args()
    args.reps = args.reps if args.reps is not None else (5 if args.mode == 'full' else 1)
    args.single_node_threads = sorted(set([1] + args.single_node_threads))
    args.core_ranks = args.core_ranks or args.hybrid_ranks * args.threads_per_node
    if min(args.hybrid_ranks, args.threads_per_node, args.core_ranks, args.reps) < 1 or args.timeout <= 0:
        parser.error('rank/thread/repetition counts and timeout must be positive')
    jobs = [] if args.mode == 'evidence' else plan(args)
    if args.mode == 'full' and not args.hybrid_mpi_arg:
        if args.dry_run:
            print('# Set --hybrid-mpi-arg options for one rank/node; see README.', file=sys.stderr)
        else:
            parser.error('full requires explicit --hybrid-mpi-arg placement; see README OpenMPI/MPICH examples')
    session = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    out = (args.output or ROOT / 'runs' / session).resolve()
    if args.dry_run:
        for i, job in enumerate(jobs, 1):
            print(shlex.join(command_for(args, job, out / 'raw' / f'{i:03d}-{job["executable"]}.csv', f'{session}-{i:03d}')))
        print(f'# {len(jobs)} commands, {args.reps} measured repetitions each; plus untimed correctness gates.')
        return 0
    out.mkdir(parents=True, exist_ok=False)
    (out / 'raw').mkdir(); (out / 'logs').mkdir(); (out / 'evidence').mkdir()
    print(f'Output: {out}', flush=True)
    env = os.environ.copy()
    env.setdefault('OMP_PROC_BIND', 'close'); env.setdefault('OMP_PLACES', 'cores')
    # This records safe controls only, never arbitrary environment contents.
    (out / 'controls.json').write_text(json.dumps(dict(
        OMP_PROC_BIND=env['OMP_PROC_BIND'], OMP_PLACES=env['OMP_PLACES'],
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}), indent=2) + '\n')
    evidence_data = evidence(args, out / 'evidence')
    if args.mode == 'evidence':
        return 0 if evidence_data['mapping_exit'] == 0 else 1
    if args.mode == 'full' and not args.allow_single_node and (evidence_data['mapped_nodes'] or 0) < 2:
        print('Need evidence of at least TWO actual nodes. Correct launcher placement/hostfile, then retry.', file=sys.stderr)
        return 1
    # Build log includes effective compiler flags. Set MPICC/CPPFLAGS/LDFLAGS
    # through make command-line or environment if local development needs them.
    build = capture(['make', '-B', '-j', str(min(4, os.cpu_count() or 1)), 'all'], out / 'build.log', 600, env)
    if build:
        print(f'Build failed ({build}); inspect {out / "build.log"}', file=sys.stderr)
        return build
    for i, job in enumerate(jobs, 1):
        raw = out / 'raw' / f'{i:03d}-{job["executable"]}-{job["experiment"]}.csv'
        command = command_for(args, job, raw, f'{session}-{i:03d}')
        log = out / 'logs' / f'{i:03d}-{job["executable"]}.log'
        print(f'[{i}/{len(jobs)}] {shlex.join(command)}', flush=True)
        env['OMP_NUM_THREADS'] = str(job['threads'])
        start = time.monotonic()
        code = capture(command, log, args.timeout, env)
        topology_error = None
        if code == 0 and job['hybrid'] and not args.allow_single_node:
            with raw.open(newline='') as f:
                wall_records = [r for r in csv.DictReader(f) if r['record'] == 'wall']
            if not wall_records or any(int(r['nodes']) != job['processes'] for r in wall_records):
                topology_error = 'hybrid requires one rank per actual node; inspect placement and raw nodes fields'
                print(topology_error, file=sys.stderr)
                code = 1
        entry = dict(index=i, command=command, job=job, exit_code=code,
                     duration_s=time.monotonic() - start, log=str(log.relative_to(out)))
        if topology_error:
            entry['topology_error'] = topology_error
        with (out / 'commands.jsonl').open('a') as f:
            f.write(json.dumps(entry) + '\n')
        if code:
            (out / 'failure.json').write_text(json.dumps(entry, indent=2) + '\n')
            print(f'FAILED ({code}); inspect {log}. No passing result is substituted.', file=sys.stderr)
            return code
    command = [sys.executable, str(ROOT / 'tools' / 'aggregate.py'), str(out / 'raw'),
               '--output', str(out / 'results.csv'), '--best', str(out / 'best.csv'),
               '--expected-reps', str(args.reps), '--require-complete']
    code = subprocess.call(command, cwd=ROOT)
    if code:
        return code
    (out / 'completed.json').write_text(json.dumps(dict(commands=len(jobs), repetitions=args.reps,
        note='Measured only on the hosts listed in evidence and raw CSV; no report conclusions inferred.'), indent=2) + '\n')
    print(f'Completed {len(jobs)} commands. Send the entire {out.name} directory back for analysis.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
