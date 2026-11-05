#!/usr/bin/env python3
"""Imaad Fazal | 23i-0656 | Section C | Assignment 3.

Analytical Q1 allocation accounting for this implementation, not an RSS measurement.
MPI/runtime/OpenMP stacks, allocator overhead and post-timing metric metadata are
not included. N=8 root checking temporarily allocates a further N*N doubles.
"""
import argparse
import csv
import struct
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, nargs="+", default=[4000, 32768])
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    if args.processes < 1 or any(n < 1 for n in args.size):
        parser.error("sizes and process count must be positive")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["kind", "N", "processes", "version", "rank", "owned_rows",
                     "matrix_bytes", "C_int_bytes", "partition_metadata_bytes", "check_extra_bytes",
                     "core_peak_bytes", "core_peak_MiB", "core_peak_GiB"])
    for n in args.size:
        maximum = (n + args.processes - 1) // args.processes
        for version in ("A", "B", "C"):
            for rank in range(args.processes):
                rows = n // args.processes + (rank < n % args.processes)
                local = rows * n
                if version == "A":
                    matrix = 2 * max(local * 8, 1) + n * n * 8
                    if rank == 0:
                        matrix += 2 * n * n * 8
                else:
                    matrix = 2 * max(local * 8, 1) + 2 * maximum * n * 8
                # Native Python/C int ABI, normally 4 bytes on cluster Linux.
                int_bytes = struct.calcsize("i")
                metadata = 2 * args.processes * int_bytes
                check_extra = 8 * n * n if rank == 0 and n == 8 else 0
                peak = matrix + metadata + check_extra
                writer.writerow(["analytical", n, args.processes, version, rank, rows,
                                 matrix, int_bytes, metadata, check_extra, peak,
                                 f"{peak / 2**20:.6f}", f"{peak / 2**30:.9f}"])


if __name__ == "__main__":
    main()
