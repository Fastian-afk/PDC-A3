/* Imaad Fazal | 23i-0656 | Section C | Assignment 3
 * Question 2: CSR propagation, collectives, halo overlap, and load balancing.
 * The production graph and feature arrays are partitioned by consecutive rows.
 */
#ifndef PDC_Q2_COMMON_H
#define PDC_Q2_COMMON_H

#include "../common/pdc.h"
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <omp.h>

enum { Q2_F = 16, Q2_T = 10, Q2_HALO = 128 };

/* Exact unsigned-32-bit generator from assignment Code 4. */
static uint32_t q2_hash32(uint32_t x)
{
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static int q2_degree(long i, long n)
{
    uint32_t h = q2_hash32((uint32_t)i);
    if (i < n / 10 && h % 4 == 0)
        return 64 + (int)(h % 64);
    return 4 + (int)(h % 9);
}

static int q2_link_offset(long i, int k)
{
    /* Unsigned arithmetic preserves the specified modulo-2^32 conversion
     * without overflowing signed long on platforms where long is 32 bits. */
    uint32_t h = q2_hash32((uint32_t)i * UINT32_C(131)
                          + (uint32_t)k * UINT32_C(7) + UINT32_C(1));
    return (int)(h % 257) - 128;
}

static double q2_initial(long i, int f)
{
    return (double)(((uint64_t)i * 31U + (unsigned)f * 17U) % 100U) / 100.0;
}

typedef struct {
    long n, start, rows;
    size_t links;
    size_t *row_ptr;
    long *col_idx;
    double *val;
    double *h, *next, *global;
    long *bounds;
    int *counts, *displs;
    int halo;
} q2_data;

static void q2_partitions(q2_data *d, int rank, int p, int balanced)
{
    d->bounds = pdc_alloc((size_t)p + 1, sizeof(long), MPI_COMM_WORLD);
    if (!balanced) {
        for (int r = 0; r < p; ++r) {
            long count;
            pdc_block(d->n, p, r, &d->bounds[r], &count);
        }
        d->bounds[p] = d->n;
    } else {
        /* COLLECTIVE: root uses degree() and a running sum, never builds CSR
         * for another rank. Clamp cuts to reserve >=128 rows per rank. */
        if (rank == 0) {
            long long total = 0, prefix = 0;
            long cursor = 0;
            for (long i = 0; i < d->n; ++i) total += q2_degree(i, d->n);
            d->bounds[0] = 0;
            for (int r = 1; r < p; ++r) {
                long lower = d->bounds[r - 1] + Q2_HALO;
                long upper = d->n - (long)(p - r) * Q2_HALO;
                long long target = total * r / p;
                while (cursor < upper && (cursor < lower || prefix < target)) {
                    prefix += q2_degree(cursor, d->n);
                    ++cursor;
                }
                d->bounds[r] = cursor;
            }
            d->bounds[p] = d->n;
        }
        MPI_Bcast(d->bounds, p + 1, MPI_LONG, 0, MPI_COMM_WORLD);
    }
    d->start = d->bounds[rank];
    d->rows = d->bounds[rank + 1] - d->start;
    if (d->rows < Q2_HALO)
        pdc_die(MPI_COMM_WORLD, "Q2 requires at least 128 users per MPI rank; increase --size or reduce -np");
    d->counts = pdc_alloc((size_t)p, sizeof(int), MPI_COMM_WORLD);
    d->displs = pdc_alloc((size_t)p, sizeof(int), MPI_COMM_WORLD);
    for (int r = 0; r < p; ++r) {
        d->counts[r] = pdc_count((size_t)(d->bounds[r + 1] - d->bounds[r]) * Q2_F, MPI_COMM_WORLD);
        d->displs[r] = pdc_count((size_t)d->bounds[r] * Q2_F, MPI_COMM_WORLD);
    }
}

static void q2_build(q2_data *d)
{
    d->row_ptr = pdc_alloc((size_t)d->rows + 1, sizeof(size_t), MPI_COMM_WORLD);
    for (long i = 0; i < d->rows; ++i) {
        size_t degree = (size_t)q2_degree(d->start + i, d->n);
        if (d->row_ptr[i] > SIZE_MAX - degree)
            pdc_die(MPI_COMM_WORLD, "Q2 CSR index overflow; reduce --size");
        d->row_ptr[i + 1] = d->row_ptr[i] + degree;
    }
    d->links = d->row_ptr[d->rows];
    d->col_idx = pdc_alloc(d->links, sizeof(long), MPI_COMM_WORLD);
    d->val = pdc_alloc(d->links, sizeof(double), MPI_COMM_WORLD);
    for (long i = 0; i < d->rows; ++i) {
        long gi = d->start + i;
        int degree = q2_degree(gi, d->n);
        for (int k = 0; k < degree; ++k) {
            size_t at = d->row_ptr[i] + (size_t)k;
            int offset = q2_link_offset(gi, k);
            /* In the halo layout the unwrapped relative index works even
             * across global row 0; no nearest-neighbour search is necessary. */
            d->col_idx[at] = d->halo ? i + Q2_HALO + offset
                                     : (gi + offset + d->n) % d->n;
            d->val[at] = 1.0 / degree;
        }
    }
    size_t feature_rows = (size_t)d->rows + (d->halo ? 2 * Q2_HALO : 0);
    d->h = pdc_alloc(feature_rows * Q2_F, sizeof(double), MPI_COMM_WORLD);
    d->next = pdc_alloc(feature_rows * Q2_F, sizeof(double), MPI_COMM_WORLD);
}

static void q2_reset(q2_data *d)
{
    long base = d->halo ? Q2_HALO : 0;
    for (long i = 0; i < d->rows; ++i)
        for (int f = 0; f < Q2_F; ++f)
            d->h[(size_t)(base + i) * Q2_F + f] = q2_initial(d->start + i, f);
}

/* OPENMP: each iteration owns one output row. Scheduling is selected with
 * omp_set_schedule; dynamic uses exactly chunk 64. MPI remains in main. */
static void q2_rows(const q2_data *d, const double *source, long begin, long end,
                    long long *thread_links)
{
    long base = d->halo ? Q2_HALO : 0;
    #pragma omp parallel for schedule(runtime)
    for (long i = begin; i < end; ++i) {
        double acc[Q2_F] = {0};
        for (size_t k = d->row_ptr[i]; k < d->row_ptr[i + 1]; ++k)
            for (int f = 0; f < Q2_F; ++f)
                acc[f] += d->val[k] * source[(size_t)d->col_idx[k] * Q2_F + f];
        for (int f = 0; f < Q2_F; ++f)
            d->next[(size_t)(base + i) * Q2_F + f] = acc[f];
        if (thread_links != NULL)
            thread_links[omp_get_thread_num()] +=
                (long long)(d->row_ptr[i + 1] - d->row_ptr[i]);
    }
}

static pdc_metrics q2_step(q2_data *d, int rank, int p, int mode,
                            int wait_first, long long *thread_links)
{
    pdc_metrics m = {0};
    for (int k = 0; k < 4; ++k) m.stage_s[k] = NAN;
    double t;
    if (mode == 1) {
        /* COLLECTIVE: all H rows, while CSR stays local. */
        t = MPI_Wtime();
        MPI_Allgatherv(d->h, d->counts[rank], MPI_DOUBLE, d->global,
                       d->counts, d->displs, MPI_DOUBLE, MPI_COMM_WORLD);
        m.comm_s += MPI_Wtime() - t;
        m.bytes_received = (long long)(d->n - d->rows) * Q2_F * (long long)sizeof(double);
        t = MPI_Wtime();
        q2_rows(d, d->global, 0, d->rows, thread_links);
        m.compute_s += MPI_Wtime() - t;
    } else if (!d->halo) {
        t = MPI_Wtime();
        q2_rows(d, d->h, 0, d->rows, thread_links);
        m.compute_s += MPI_Wtime() - t;
    } else {
        MPI_Request request[4];
        if (p == 1) {
            /* A single-rank circular graph needs local copies, no messages. */
            t = MPI_Wtime();
            memcpy(d->h, d->h + (size_t)d->rows * Q2_F,
                   (size_t)Q2_HALO * Q2_F * sizeof(double));
            memcpy(d->h + (size_t)(Q2_HALO + d->rows) * Q2_F,
                   d->h + (size_t)Q2_HALO * Q2_F,
                   (size_t)Q2_HALO * Q2_F * sizeof(double));
            m.compute_s += MPI_Wtime() - t;
        } else {
            int left = (rank - 1 + p) % p, right = (rank + 1) % p;
            int count = Q2_HALO * Q2_F;
            t = MPI_Wtime();
            /* POINT TO POINT: distinct tags also distinguish both directions
             * when left == right (p=2). Send buffers are read-only. */
            MPI_Irecv(d->h, count, MPI_DOUBLE, left, 201, MPI_COMM_WORLD, &request[0]);
            MPI_Irecv(d->h + (size_t)(Q2_HALO + d->rows) * Q2_F,
                      count, MPI_DOUBLE, right, 202, MPI_COMM_WORLD, &request[1]);
            MPI_Isend(d->h + (size_t)Q2_HALO * Q2_F,
                      count, MPI_DOUBLE, left, 202, MPI_COMM_WORLD, &request[2]);
            MPI_Isend(d->h + (size_t)d->rows * Q2_F,
                      count, MPI_DOUBLE, right, 201, MPI_COMM_WORLD, &request[3]);
            m.comm_s += MPI_Wtime() - t;
            m.bytes_received = 2LL * Q2_HALO * Q2_F * (long long)sizeof(double);
        }
        if (!wait_first && d->rows > 2 * Q2_HALO) {
            /* OVERLAP: rows whose dependencies are entirely local. */
            t = MPI_Wtime();
            q2_rows(d, d->h, Q2_HALO, d->rows - Q2_HALO, thread_links);
            m.compute_s += MPI_Wtime() - t;
        }
        if (p > 1) {
            t = MPI_Wtime();
            MPI_Waitall(4, request, MPI_STATUSES_IGNORE);
            m.wait_s = MPI_Wtime() - t;
            m.comm_s += m.wait_s;
        }
        t = MPI_Wtime();
        if (wait_first || d->rows <= 2 * Q2_HALO) {
            /* No interior if <=256 local rows; compute boundaries once. */
            q2_rows(d, d->h, 0, d->rows, thread_links);
        } else {
            q2_rows(d, d->h, 0, Q2_HALO, thread_links);
            q2_rows(d, d->h, d->rows - Q2_HALO, d->rows, thread_links);
        }
        m.compute_s += MPI_Wtime() - t;
    }
    double *swap = d->h; d->h = d->next; d->next = swap;
    return m;
}

/* Independent scalar oracle: a feature of a row after t steps depends only
 * on rows within distance 128*t. Shrink a periodic strip one halo per step.
 * This avoids replicating the production H/CSR just to validate samples. */
static void q2_oracle(long n, const long rows[3], const int features[3],
                       double expected[Q2_T][3])
{
    enum { RADIUS = Q2_HALO * Q2_T, WIDTH = 2 * RADIUS + 1 };
    double current[WIDTH], next[WIDTH];
    for (int sample = 0; sample < 3; ++sample) {
        for (int j = 0; j < WIDTH; ++j) {
            long row = (rows[sample] + j - RADIUS) % n;
            if (row < 0) row += n;
            current[j] = q2_initial(row, features[sample]);
        }
        for (int step = 0; step < Q2_T; ++step) {
            int margin = (step + 1) * Q2_HALO;
            for (int j = margin; j < WIDTH - margin; ++j) {
                long row = (rows[sample] + j - RADIUS) % n;
                if (row < 0) row += n;
                int degree = q2_degree(row, n);
                double acc = 0, weight = 1.0 / degree;
                for (int k = 0; k < degree; ++k)
                    acc += weight * current[j + q2_link_offset(row, k)];
                next[j] = acc;
            }
            for (int j = margin; j < WIDTH - margin; ++j) current[j] = next[j];
            expected[step][sample] = current[RADIUS];
        }
    }
}

/* Tiny full-array oracle is deliberately separate from the production CSR
 * and uses on-the-fly link generation. Only rank 0 owns these validation
 * buffers; they are never part of the timed algorithm. */
static void q2_reference_step(long n, double *h, double *next)
{
    for (long i = 0; i < n; ++i) {
        int degree = q2_degree(i, n);
        double weight = 1.0 / degree;
        for (int f = 0; f < Q2_F; ++f) {
            double acc = 0;
            for (int k = 0; k < degree; ++k) {
                long j = (i + q2_link_offset(i, k) + n) % n;
                acc += weight * h[(size_t)j * Q2_F + f];
            }
            next[(size_t)i * Q2_F + f] = acc;
        }
    }
}

static int q2_check(q2_data *d, int rank, int p, const long samples[3],
                     const int features[3], const double expected[3],
                     const double *full_reference, double *gathered, int printing)
{
    double local[3] = {0}, global[3] = {0};
    int good = 1;
    long base = d->halo ? Q2_HALO : 0;
    for (int s = 0; s < 3; ++s)
        if (samples[s] >= d->start && samples[s] < d->start + d->rows) {
            local[s] = d->h[(size_t)(samples[s] - d->start + base) * Q2_F + features[s]];
            if (!isfinite(local[s]) || fabs(local[s] - expected[s]) > 2e-13) good = 0;
        }
    MPI_Reduce(local, global, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (d->n <= 4096) {
        MPI_Gatherv(d->h + (size_t)base * Q2_F, d->counts[rank], MPI_DOUBLE,
                    gathered, d->counts, d->displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0)
            for (size_t i = 0; i < (size_t)d->n * Q2_F; ++i)
                if (!isfinite(gathered[i]) || fabs(gathered[i] - full_reference[i]) > 2e-13) {
                    fprintf(stderr, "Q2 full-array mismatch at flat index %zu: %.17g != %.17g\n",
                            i, gathered[i], full_reference[i]);
                    good = 0; break;
                }
    }
    if (rank == 0 && (d->n == 100000 || d->n == 1000000) && printing) {
        static const double table[2][3] = {
            {0.493896795310, 0.497876802886, 0.497973641990},
            {0.500882792529, 0.499154198420, 0.492279183027}
        };
        int row = d->n == 100000 ? 0 : 1;
        for (int s = 0; s < 3; ++s)
            if (fabs(global[s] - table[row][s]) > 5.05e-13) good = 0;
    }
    int all_good;
    MPI_Allreduce(&good, &all_good, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_good) {
        if (rank == 0) fprintf(stderr, "Q2 correctness failure (n=%ld).\n", d->n);
        return 0;
    }
    if (printing) {
        for (int r = 0; r < p; ++r) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == r) {
                for (int s = 0; s < 3; ++s)
                    if (samples[s] >= d->start && samples[s] < d->start + d->rows)
                        printf("rank %d H(%ld,%d) = %.12f\n", rank, samples[s], features[s], local[s]);
                fflush(stdout);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    return 1;
}

static int q2_main(int argc, char **argv, int mode)
{
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    pdc_options opt = pdc_parse(argc, argv, 100000);
    if (mode == 0 && p != 1)
        pdc_die(MPI_COMM_WORLD, "Q2 OpenMP-only executable requires -np 1");
    if (opt.size < (long)p * Q2_HALO)
        pdc_die(MPI_COMM_WORLD, "Q2 --size must be >=128 times the number of MPI ranks");
    if (opt.size > INT_MAX / Q2_F)
        pdc_die(MPI_COMM_WORLD, "Q2 --size exceeds portable MPI int-count/displacement range");
    if (strcmp(opt.schedule, "static") != 0 && strcmp(opt.schedule, "dynamic") != 0)
        pdc_die(MPI_COMM_WORLD, "Q2 --schedule must be static or dynamic");
    if (strcmp(opt.partition, "rows") != 0 && strcmp(opt.partition, "links") != 0
        && strcmp(opt.partition, "default") != 0)
        pdc_die(MPI_COMM_WORLD, "Q2 --partition must be rows or links");
    int partition_explicit = 0;
    for (int a = 1; a < argc; ++a)
        if (strcmp(argv[a], "--partition") == 0) partition_explicit = 1;
    if (mode == 3 && !partition_explicit) opt.partition = "links";
    int balanced = mode == 3;
    if (strcmp(opt.partition, "rows") == 0) balanced = 0;
    if (strcmp(opt.partition, "links") == 0) balanced = 1;
    omp_set_dynamic(0);
    omp_set_num_threads(opt.threads);
    omp_set_schedule(strcmp(opt.schedule, "dynamic") == 0 ? omp_sched_dynamic : omp_sched_static,
                     strcmp(opt.schedule, "dynamic") == 0 ? 64 : 0);
    int actual_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    int minimum_threads, maximum_threads;
    MPI_Allreduce(&actual_threads, &minimum_threads, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&actual_threads, &maximum_threads, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (minimum_threads != maximum_threads)
        pdc_die(MPI_COMM_WORLD, "Q2 actual OpenMP team sizes differ across ranks; check runtime limits");
    opt.threads = actual_threads;
    const char *version = mode == 0 ? "omp" : mode == 1 ? "allgatherv" : mode == 2 ? "halo" : "balanced";
    q2_data d = {0};
    d.n = opt.size;
    d.halo = mode >= 2;
    q2_partitions(&d, rank, p, balanced);
    q2_build(&d);
    if (mode == 1) d.global = pdc_alloc((size_t)d.n * Q2_F, sizeof(double), MPI_COMM_WORLD);
    unsigned long long local_links = (unsigned long long)d.links, total_links = 0;
    MPI_Allreduce(&local_links, &total_links, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if ((d.n == 100000 && total_links != 1020227ULL)
        || (d.n == 1000000 && total_links != 10160271ULL))
        pdc_die(MPI_COMM_WORLD, "Q2 graph edge count does not match Table 3");
    if (rank == 0)
        printf("Q2 %s n=%ld ranks=%d threads=%d partition=%s schedule=%s wait_first=%d links=%llu\n",
               version, d.n, p, opt.threads, balanced ? "links" : "rows", opt.schedule,
               opt.wait_first, total_links);
    for (int r = 0; r < p; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r) {
            printf("rank %d rows=[%ld,%ld) local_links=%zu\n", rank, d.start, d.start + d.rows, d.links);
            fflush(stdout);
        }
    }
    const long samples[3] = {0, d.n > 12345 ? 12345 : d.n / 2, d.n - 1};
    const int features[3] = {0, 7, 15};
    double expected[Q2_T][3];
    q2_oracle(d.n, samples, features, expected);
    double *reference = NULL, *reference_next = NULL, *gathered = NULL;
    if (rank == 0 && d.n <= 4096) {
        reference = pdc_alloc((size_t)d.n * Q2_F, sizeof(double), MPI_COMM_WORLD);
        reference_next = pdc_alloc((size_t)d.n * Q2_F, sizeof(double), MPI_COMM_WORLD);
        gathered = pdc_alloc((size_t)d.n * Q2_F, sizeof(double), MPI_COMM_WORLD);
        for (long i = 0; i < d.n; ++i)
            for (int f = 0; f < Q2_F; ++f)
                reference[(size_t)i * Q2_F + f] = q2_initial(i, f);
    }
    long long *thread_links = pdc_alloc((size_t)opt.threads, sizeof(long long), MPI_COMM_WORLD);
    q2_reset(&d);
    /* Untimed correctness gate checks every step before reporting timings. */
    for (int step = 0; step < Q2_T; ++step) {
        (void)q2_step(&d, rank, p, mode, opt.wait_first, step == 0 ? thread_links : NULL);
        if (rank == 0 && d.n <= 4096) {
            q2_reference_step(d.n, reference, reference_next);
            double *swap = reference; reference = reference_next; reference_next = swap;
        }
        if (!q2_check(&d, rank, p, samples, features, expected[step], reference, gathered,
                      step == Q2_T - 1))
            pdc_die(MPI_COMM_WORLD, "Q2 untimed correctness gate failed");
    }
    if (strcmp(opt.schedule, "static") == 0) {
        for (int r = 0; r < p; ++r) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == r) {
                for (int t = 0; t < opt.threads; ++t)
                    printf("rank %d thread %d static_links_per_step=%lld\n", rank, t, thread_links[t]);
                fflush(stdout);
            }
        }
    }
    for (int repetition = 0; repetition < opt.reps; ++repetition) {
        q2_reset(&d);
        pdc_metrics per_step[Q2_T], total = {0};
        double step_elapsed[Q2_T];
        for (int k = 0; k < 4; ++k) total.stage_s[k] = NAN;
        MPI_Barrier(MPI_COMM_WORLD);
        double begin = MPI_Wtime();
        for (int step = 0; step < Q2_T; ++step) {
            double t = MPI_Wtime();
            per_step[step] = q2_step(&d, rank, p, mode, opt.wait_first, NULL);
            step_elapsed[step] = MPI_Wtime() - t;
            total.compute_s += per_step[step].compute_s;
            total.comm_s += per_step[step].comm_s;
            total.wait_s += per_step[step].wait_s;
            total.bytes_received += per_step[step].bytes_received;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed = MPI_Wtime() - begin;
        int correct = q2_check(&d, rank, p, samples, features, expected[Q2_T - 1],
                               reference, gathered, 0);
        pdc_record(MPI_COMM_WORLD, &opt, "Q2", version, repetition, -1, Q2_T, elapsed, &total, correct);
        for (int step = 0; step < Q2_T; ++step)
            pdc_record(MPI_COMM_WORLD, &opt, "Q2", version, repetition, step, Q2_T,
                       step_elapsed[step], &per_step[step], correct);
        if (!correct) pdc_die(MPI_COMM_WORLD, "Q2 measured result failed validation");
    }
    free(thread_links); free(reference); free(reference_next); free(gathered);
    free(d.row_ptr); free(d.col_idx); free(d.val);
    free(d.h); free(d.next); free(d.global);
    free(d.bounds); free(d.counts); free(d.displs);
    MPI_Finalize();
    return 0;
}

#endif
