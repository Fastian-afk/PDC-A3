/* Imaad Fazal | 23i-0656 | Section C | CS3006 Assignment 3
 * Shared implementation for Task 3 versions 1--3.
 * MPI calls are made only by the main thread, outside OpenMP regions.
 */
#ifndef PDC_TASK3_COMMON_H
#define PDC_TASK3_COMMON_H
#include "../common/pdc.h"
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#define Q3_LAYERS 8
#define Q3_STEPS 5
#define Q3_BATCH 512

#ifndef Q3_VERSION
#error "Define Q3_VERSION as 1 (Reduce+Bcast), 2 (Allreduce), or 3 (ring)."
#endif

static const char *q3_version_name(void)
{
    return Q3_VERSION == 1 ? "reduce-bcast" : Q3_VERSION == 2 ? "allreduce" : "ring";
}

static size_t q3_product(size_t a, size_t b, MPI_Comm comm)
{
    if (b != 0 && a > SIZE_MAX / b)
        pdc_die(comm, "Q3 size arithmetic overflow: choose a smaller --size.");
    return a * b;
}

/* Reused workspace avoids including allocation in the measured all-reduce.
 * Preparation is exercised by the mandatory untimed correctness gate.
 * A process must not invoke ring_allreduce concurrently from multiple threads.
 */
static double *q3_ring_tmp = NULL;
static size_t q3_ring_capacity = 0;
/* Per-call counters distinguish local vector addition from MPI-call time.
 * Exposed Waitall time is a subset of q3_last_comm_s, never added twice. */
static double q3_last_comm_s = 0.0;
static double q3_last_wait_s = 0.0;
static double q3_last_add_s = 0.0;

static void q3_ring_prepare(long n, int p, MPI_Comm comm)
{
    size_t need;
    if (n < 0) pdc_die(comm, "ring_allreduce: vector length must be nonnegative.");
    need = p == 1 ? 0 : (size_t)(n / p + (n % p != 0));
    (void)pdc_count(need, comm);
    if (need > q3_ring_capacity || q3_ring_tmp == NULL) {
        double *replacement = pdc_alloc(need, sizeof(double), comm);
        free(q3_ring_tmp);
        q3_ring_tmp = replacement;
        q3_ring_capacity = need;
    }
}

/* Task 3.2: an uneven-chunk ring all-reduce, with no collective data movement.
 * Counts of zero are legal; p=1 needs no messages.  No in-flight send buffer
 * is modified.  Every receive completes before its data is read or overwritten.
 */
void ring_allreduce(double *v, long n, MPI_Comm comm)
{
    int rank, p, left, right;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);
    q3_ring_prepare(n, p, comm);
    q3_last_comm_s = q3_last_wait_s = q3_last_add_s = 0.0;
    if (p == 1) return;
    left = (rank - 1 + p) % p;
    right = (rank + 1) % p;
    /* POINT-TO-POINT: reduce-scatter.  Rank r ends with chunk (r+1)%p. */
    for (int s = 0; s < p - 1; ++s) {
        int send_chunk = (rank - s + p) % p;
        int recv_chunk = (rank - s - 1 + p) % p;
        long send_start, send_n, recv_start, recv_n;
        MPI_Request req[2];
        pdc_block(n, p, send_chunk, &send_start, &send_n);
        pdc_block(n, p, recv_chunk, &recv_start, &recv_n);
        double post_begin = MPI_Wtime();
        MPI_Irecv(q3_ring_tmp, pdc_count((size_t)recv_n, comm), MPI_DOUBLE,
                  left, 310, comm, &req[0]);
        MPI_Isend(v + send_start, pdc_count((size_t)send_n, comm), MPI_DOUBLE,
                  right, 310, comm, &req[1]);
        double post_end = MPI_Wtime();
        MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
        double waited = MPI_Wtime();
        q3_last_comm_s += waited - post_begin;
        q3_last_wait_s += waited - post_end;
        for (long j = 0; j < recv_n; ++j)
            v[recv_start + j] += q3_ring_tmp[j];
        q3_last_add_s += MPI_Wtime() - waited;
    }
    /* POINT-TO-POINT: allgather the completely reduced chunks. */
    for (int s = 0; s < p - 1; ++s) {
        int send_chunk = (rank + 1 - s + p) % p;
        int recv_chunk = (rank - s + p) % p;
        long send_start, send_n, recv_start, recv_n;
        MPI_Request req[2];
        pdc_block(n, p, send_chunk, &send_start, &send_n);
        pdc_block(n, p, recv_chunk, &recv_start, &recv_n);
        double post_begin = MPI_Wtime();
        MPI_Irecv(v + recv_start, pdc_count((size_t)recv_n, comm), MPI_DOUBLE,
                  left, 311, comm, &req[0]);
        MPI_Isend(v + send_start, pdc_count((size_t)send_n, comm), MPI_DOUBLE,
                  right, 311, comm, &req[1]);
        double post_end = MPI_Wtime();
        MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
        double waited = MPI_Wtime();
        q3_last_comm_s += waited - post_begin;
        q3_last_wait_s += waited - post_end;
    }
}

static long long q3_ring_received(long n, int rank, int p, MPI_Comm comm)
{
    long unused, a, b;
    if (p == 1) return 0;
    pdc_block(n, p, rank, &unused, &a);
    pdc_block(n, p, (rank + 1) % p, &unused, &b);
    if ((unsigned long long)n > (unsigned long long)LLONG_MAX / 16ULL)
        pdc_die(comm, "Q3 byte accounting overflows a signed 64-bit integer.");
    return (2LL * n - a - b) * (long long)sizeof(double);
}

static void q3_reduce(double *v, long n, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    q3_last_comm_s = q3_last_add_s = 0.0;
    q3_last_wait_s = NAN;
    double begin = MPI_Wtime();
    if (Q3_VERSION == 3) {
        ring_allreduce(v, n, comm);
    } else if (Q3_VERSION == 2) {
        MPI_Allreduce(MPI_IN_PLACE, v, pdc_count((size_t)n, comm), MPI_DOUBLE,
                      MPI_SUM, comm);
    } else {
        /* COLLECTIVES: root's input already occupies its receive buffer. */
        if (rank == 0)
            MPI_Reduce(MPI_IN_PLACE, v, pdc_count((size_t)n, comm), MPI_DOUBLE,
                       MPI_SUM, 0, comm);
        else
            MPI_Reduce(v, NULL, pdc_count((size_t)n, comm), MPI_DOUBLE,
                       MPI_SUM, 0, comm);
        MPI_Bcast(v, pdc_count((size_t)n, comm), MPI_DOUBLE, 0, comm);
    }
    if (Q3_VERSION != 3) q3_last_comm_s = MPI_Wtime() - begin;
}

static pdc_metrics q3_empty_metrics(void)
{
    pdc_metrics m = {0};
    for (int j = 0; j < 4; ++j) m.stage_s[j] = NAN;
    return m;
}

static void q3_require(int local_ok, MPI_Comm comm, const char *message)
{
    int all_ok;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
    if (!all_ok) pdc_die(comm, message);
}

static double q3_micro_input(long i, int rank)
{
    /* Exactly representable dyadic values, different on every rank. */
    return (double)(rank + 1) + (double)(i % 17) / 16.0;
}

static void q3_check_ring(MPI_Comm comm)
{
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);
    const long sizes[] = {0, 1, p - 1, p, p + 3, 17, 129};
    for (size_t t = 0; t < sizeof(sizes) / sizeof(sizes[0]); ++t) {
        long n = sizes[t];
        double *actual = pdc_alloc((size_t)n, sizeof(double), comm);
        double *reference = pdc_alloc((size_t)n, sizeof(double), comm);
        for (long i = 0; i < n; ++i) actual[i] = q3_micro_input(i, rank);
        MPI_Allreduce(actual, reference, pdc_count((size_t)n, comm), MPI_DOUBLE,
                      MPI_SUM, comm);
        ring_allreduce(actual, n, comm);
        int good = 1;
        for (long i = 0; i < n; ++i)
            if (actual[i] != reference[i]) good = 0;
        free(reference);
        free(actual);
        q3_require(good, comm, "Q3 ring self-test disagrees with MPI_Allreduce.");
    }
    if (rank == 0)
        printf("Q3 ring vs MPI_Allreduce: PASS (zero, tiny, uneven chunks).\n");
}

static void q3_fill_micro(double *v, long n, int rank)
{
    for (long i = 0; i < n; ++i) v[i] = q3_micro_input(i, rank);
}

static void q3_validate_micro(const double *v, long n, int p, MPI_Comm comm)
{
    int good = 1;
    double base = (double)p * (p + 1) / 2.0;
    for (long i = 0; i < n; ++i)
        if (v[i] != base + (double)p * (double)(i % 17) / 16.0) good = 0;
    q3_require(good, comm, "Q3 all-reduce microbenchmark failed correctness.");
}

static int q3_micro(const pdc_options *o, MPI_Comm comm)
{
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);
    if (o->bytes < 0 || o->bytes % (long)sizeof(double) != 0)
        pdc_die(comm, "Q3 --micro requires --bytes >= 0 and divisible by 8.");
    long n = o->bytes / (long)sizeof(double);
    (void)pdc_count((size_t)n, comm);
    double *v = pdc_alloc((size_t)n, sizeof(double), comm);
    pdc_options csv_options = *o;
    csv_options.size = n; /* Micro CSV size is a count of doubles, not bytes. */
    q3_fill_micro(v, n, rank);
    q3_reduce(v, n, comm); /* Untimed warmup and correctness gate. */
    q3_validate_micro(v, n, p, comm);
    if (rank == 0)
        printf("Q3 micro %s: %ld bytes (%ld doubles), %d ranks, correctness PASS\n",
               q3_version_name(), o->bytes, n, p);
    for (int rep = 1; rep <= o->reps; ++rep) {
        pdc_metrics metrics = q3_empty_metrics();
        q3_fill_micro(v, n, rank);
        MPI_Barrier(comm);
        double begin = MPI_Wtime();
        q3_reduce(v, n, comm);
        MPI_Barrier(comm);
        double elapsed = MPI_Wtime() - begin;
        metrics.comm_s = q3_last_comm_s;
        metrics.compute_s = q3_last_add_s;
        metrics.wait_s = q3_last_wait_s;
        metrics.bytes_received = Q3_VERSION == 3 ? q3_ring_received(n, rank, p, comm) : -1;
        q3_validate_micro(v, n, p, comm);
        pdc_record(comm, &csv_options, "Q3micro", q3_version_name(), rep,
                   -1, 1, elapsed, &metrics, 1);
    }
    free(v);
    return 0;
}

/* These tables are an INDEPENDENT validator only.  They never generate a
 * computed gradient or weight.  Formula periods are nine in i and seven in j,
 * so 8*9*7*5*512 scalar products suffice to validate every matrix entry.
 */
static long q3_expected_g[Q3_STEPS][Q3_LAYERS][9][7];
static long q3_expected_sum[Q3_LAYERS][9][7];

static void q3_prepare_reference(void)
{
    memset(q3_expected_sum, 0, sizeof(q3_expected_sum));
    for (int s = 0; s < Q3_STEPS; ++s)
        for (int l = 0; l < Q3_LAYERS; ++l)
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 7; ++j) {
                    long gsum = 0;
                    for (int g = 0; g < Q3_BATCH; ++g) {
                        int x = (i + 3 * g + 5 * l + s) % 9 - 4;
                        int e = (2 * g + j + l + 3 * s) % 7 - 3;
                        gsum += x * e;
                    }
                    q3_expected_g[s][l][i][j] = gsum;
                    q3_expected_sum[l][i][j] += gsum;
                }
}

static void q3_validate_gradient(const double *grad, long m, int step, MPI_Comm comm)
{
    int good = 1;
    for (int l = 0; l < Q3_LAYERS; ++l)
        for (long i = 0; i < m; ++i)
            for (long j = 0; j < m; ++j) {
                size_t at = ((size_t)l * (size_t)m + (size_t)i) * (size_t)m + (size_t)j;
                if (grad[at] != (double)q3_expected_g[step][l][i % 9][j % 7]) good = 0;
            }
    q3_require(good, comm, "Q3 gradient differs from independent global-batch reference.");
}

static void q3_validate_weights(const double *w, long m, MPI_Comm comm)
{
    int good = 1;
    for (int l = 0; l < Q3_LAYERS; ++l)
        for (long i = 0; i < m; ++i)
            for (long j = 0; j < m; ++j) {
                size_t at = ((size_t)l * (size_t)m + (size_t)i) * (size_t)m + (size_t)j;
                double expected = -(double)q3_expected_sum[l][i % 9][j % 7] / 65536.0;
                if (w[at] != expected) good = 0;
            }
    /* Every entry on every rank must agree with the same exact reference.
     * This is stronger than merely comparing three printed weights. */
    q3_require(good, comm, "Q3 final weights differ across ranks or from the exact reference.");
}

static void q3_make_inputs(double *x, double *e, long m, long first, long local)
{
    /* Inputs for all five steps are prepared OUTSIDE every measured region.
     * Only the rank's contiguous examples are constructed and retained.
     */
    for (int s = 0; s < Q3_STEPS; ++s)
        for (int l = 0; l < Q3_LAYERS; ++l) {
            size_t base = ((size_t)s * Q3_LAYERS + (size_t)l) * (size_t)m * (size_t)local;
            for (long i = 0; i < m; ++i)
                for (long g = 0; g < local; ++g)
                    x[base + (size_t)i * (size_t)local + (size_t)g] =
                        (double)((i % 9 + 3 * ((first + g) % 9) + 5 * l + s) % 9 - 4);
            for (long g = 0; g < local; ++g)
                for (long j = 0; j < m; ++j)
                    e[base + (size_t)g * (size_t)m + (size_t)j] =
                        (double)((2 * ((first + g) % 7) + j % 7 + l + 3 * s) % 7 - 3);
        }
}

static void q3_multiply(const double *x, const double *e, double *gradient,
                        long m, long local, int step)
{
    /* OPENMP: each output row belongs exclusively to one thread.
     * i-k-j order gives contiguous E and gradient accesses in the inner loop.
     * No MPI call occurs inside this parallel region.
     */
    #pragma omp parallel for schedule(static)
    for (long row = 0; row < Q3_LAYERS * m; ++row) {
        long layer = row / m;
        long i = row % m;
        size_t input_base = ((size_t)step * Q3_LAYERS + (size_t)layer) * (size_t)m * (size_t)local;
        double *out = gradient + (size_t)row * (size_t)m;
        for (long j = 0; j < m; ++j) out[j] = 0.0;
        for (long g = 0; g < local; ++g) {
            double a = x[input_base + (size_t)i * (size_t)local + (size_t)g];
            const double *erow = e + input_base + (size_t)g * (size_t)m;
            for (long j = 0; j < m; ++j) out[j] += a * erow[j];
        }
    }
}

static void q3_update(double *weights, const double *gradient, size_t count)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) weights[i] -= gradient[i] / 65536.0;
}

static void q3_capture(const double *gradient, long m, double captured[3])
{
    long i = m > 1 ? 1 : 0;
    long j = m > 2 ? 2 : m - 1;
    captured[0] = gradient[(size_t)i * (size_t)m + (size_t)j];
    captured[1] = gradient[(size_t)m * (size_t)m - 1];
    captured[2] = gradient[(size_t)Q3_LAYERS * (size_t)m * (size_t)m - 1];
}

static void q3_check_captured(const double captured[3], long m, MPI_Comm comm)
{
    long i = m > 1 ? 1 : 0;
    long j = m > 2 ? 2 : m - 1;
    int good = captured[0] == (double)q3_expected_g[0][0][i % 9][j % 7] &&
               captured[1] == (double)q3_expected_g[0][0][(m - 1) % 9][(m - 1) % 7] &&
               captured[2] == (double)q3_expected_g[0][7][(m - 1) % 9][(m - 1) % 7];
    if (m == 512)
        good = good && captured[0] == 33 && captured[1] == 24 && captured[2] == 27;
    if (m == 1024)
        good = good && captured[0] == 33 && captured[1] == 5 && captured[2] == -20;
    q3_require(good, comm, "Q3 step-0 gradient samples do not match Table 4(a).");
}

static void q3_print_results(const double *w, const double captured[3], long m,
                             int rank, int p, MPI_Comm comm)
{
    if (rank == 0) {
        long i = m > 1 ? 1 : 0;
        long j = m > 2 ? 2 : m - 1;
        printf("Q3 %s m=%ld: G0(%ld,%ld)=%.16g G0(%ld,%ld)=%.16g G7(%ld,%ld)=%.16g\n",
               q3_version_name(), m, i, j, captured[0], m - 1, m - 1, captured[1],
               m - 1, m - 1, captured[2]);
        fflush(stdout);
    }
    /* Serialize PRINTING only, outside all measured regions. */
    for (int r = 0; r < p; ++r) {
        MPI_Barrier(comm);
        if (rank == r) {
            long i = m > 1 ? 1 : 0;
            long j = m > 2 ? 2 : m - 1;
            printf("rank %d final W0(0,0)=%.16g W0(%ld,%ld)=%.16g W7(%ld,%ld)=%.16g correctness=PASS\n",
                   rank, w[0], i, j, w[(size_t)i * (size_t)m + (size_t)j],
                   m - 1, m - 1,
                   w[(size_t)Q3_LAYERS * (size_t)m * (size_t)m - 1]);
            fflush(stdout);
        }
    }
    MPI_Barrier(comm);
}

static int q3_training(const pdc_options *o, MPI_Comm comm)
{
    int rank, p;
    long first, local;
    long m = o->size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);
    if (m < 1) pdc_die(comm, "Q3 --size m must be positive.");
    size_t cells = q3_product(Q3_LAYERS, q3_product((size_t)m, (size_t)m, comm), comm);
    if (cells > (size_t)LONG_MAX)
        pdc_die(comm, "Q3 gradient vector exceeds LONG_MAX elements.");
    (void)pdc_count(cells, comm);
    pdc_block(Q3_BATCH, p, rank, &first, &local);
    size_t input_cells = q3_product(Q3_STEPS * Q3_LAYERS,
                                    q3_product((size_t)m, (size_t)local, comm), comm);
    double *x = pdc_alloc(input_cells, sizeof(double), comm);
    double *e = pdc_alloc(input_cells, sizeof(double), comm);
    double *grad = pdc_alloc(cells, sizeof(double), comm);
    double *weights = pdc_alloc(cells, sizeof(double), comm);
    q3_make_inputs(x, e, m, first, local);
    q3_prepare_reference();
    if (Q3_VERSION == 3) q3_ring_prepare((long)cells, p, comm);
    if (rank == 0) {
        printf("Q3 %s: m=%ld layers=8 batch=512 steps=5 ranks=%d threads=%d\n",
               q3_version_name(), m, p, o->threads);
        printf("Local X/E contain only owned examples for all 5 steps; all-rank weights/gradients are required replicas.\n");
    }
    /* Mandatory untimed gate: every element of all five reduced gradients,
     * and every final weight, is independently checked on every rank.
     */
    double captured[3] = {0};
    for (int s = 0; s < Q3_STEPS; ++s) {
        q3_multiply(x, e, grad, m, local, s);
        q3_reduce(grad, (long)cells, comm);
        q3_validate_gradient(grad, m, s, comm);
        if (s == 0) q3_capture(grad, m, captured);
        q3_update(weights, grad, cells);
    }
    q3_validate_weights(weights, m, comm);
    q3_check_captured(captured, m, comm);
    if (rank == 0) printf("Q3 untimed full-gradient/final-weight correctness gate: PASS\n");
    for (int rep = 1; rep <= o->reps; ++rep) {
        pdc_metrics per_step[Q3_STEPS];
        pdc_metrics total = q3_empty_metrics();
        double step_elapsed[Q3_STEPS];
        memset(weights, 0, cells * sizeof(double));
        long long per_step_bytes = Q3_VERSION == 3 ?
            q3_ring_received((long)cells, rank, p, comm) : -1;
        total.bytes_received = per_step_bytes < 0 ? -1 : Q3_STEPS * per_step_bytes;
        total.wait_s = Q3_VERSION == 3 ? 0.0 : NAN;
        MPI_Barrier(comm);
        double begin = MPI_Wtime();
        for (int s = 0; s < Q3_STEPS; ++s) {
            per_step[s] = q3_empty_metrics();
            per_step[s].wait_s = Q3_VERSION == 3 ? 0.0 : NAN;
            per_step[s].bytes_received = per_step_bytes;
            double t0 = MPI_Wtime();
            q3_multiply(x, e, grad, m, local, s);
            double t1 = MPI_Wtime();
            q3_reduce(grad, (long)cells, comm);
            double t2 = MPI_Wtime();
            if (s == 0) q3_capture(grad, m, captured);
            q3_update(weights, grad, cells);
            double t3 = MPI_Wtime();
            per_step[s].compute_s = (t1 - t0) + (t3 - t2) + q3_last_add_s;
            per_step[s].comm_s = q3_last_comm_s;
            per_step[s].wait_s = q3_last_wait_s;
            step_elapsed[s] = t3 - t0;
            total.compute_s += per_step[s].compute_s;
            total.comm_s += per_step[s].comm_s;
            if (Q3_VERSION == 3) total.wait_s += per_step[s].wait_s;
        }
        MPI_Barrier(comm);
        double elapsed = MPI_Wtime() - begin;
        q3_validate_weights(weights, m, comm);
        q3_validate_gradient(grad, m, Q3_STEPS - 1, comm);
        q3_check_captured(captured, m, comm);
        for (int s = 0; s < Q3_STEPS; ++s)
            pdc_record(comm, o, "Q3", q3_version_name(), rep, s, Q3_STEPS,
                       step_elapsed[s], &per_step[s], 1);
        pdc_record(comm, o, "Q3", q3_version_name(), rep, -1, Q3_STEPS,
                   elapsed, &total, 1);
    }
    q3_print_results(weights, captured, m, rank, p, comm);
    free(weights);
    free(grad);
    free(e);
    free(x);
    return 0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    pdc_options options = pdc_parse(argc, argv, 512);
    omp_set_dynamic(0);
    omp_set_num_threads(options.threads);
    int actual_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    int minimum_threads, maximum_threads;
    MPI_Allreduce(&actual_threads, &minimum_threads, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&actual_threads, &maximum_threads, 1, MPI_INT, MPI_MAX, comm);
    if (minimum_threads != maximum_threads)
        pdc_die(comm, "Q3 actual OpenMP team sizes differ across ranks; check OMP_THREAD_LIMIT and launcher environment.");
    options.threads = actual_threads;
    if (Q3_VERSION == 3) q3_check_ring(comm);
    int result = options.micro ? q3_micro(&options, comm) : q3_training(&options, comm);
    free(q3_ring_tmp);
    q3_ring_tmp = NULL;
    MPI_Finalize();
    return result;
}
#endif
