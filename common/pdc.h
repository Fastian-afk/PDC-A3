/* Imaad Fazal | 23i-0656 | Section C | Assignment 3
 * Shared command-line, allocation and measurement utilities. */
#ifndef PDC_COMMON_H
#define PDC_COMMON_H

#include <mpi.h>
#include <omp.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    long size;
    int threads, reps, validate, wait_first, micro;
    long bytes;
    const char *csv, *run_id, *experiment, *schedule, *partition;
} pdc_options;

typedef struct {
    double compute_s, comm_s, wait_s, stage_s[4];
    long long bytes_received; /* -1: algorithm-selected collective volume unknown */
} pdc_metrics;

static inline void pdc_die(MPI_Comm comm, const char *message)
{
    int initialized = 0, rank = 0;
    MPI_Initialized(&initialized);
    if (initialized) MPI_Comm_rank(comm, &rank);
    fprintf(stderr, "ERROR rank %d: %s\n", rank, message);
    if (initialized) MPI_Abort(comm, EXIT_FAILURE);
    exit(EXIT_FAILURE);
}

static inline size_t pdc_product(size_t a, size_t b, MPI_Comm comm)
{
    if (a && b > SIZE_MAX / a) pdc_die(comm, "size arithmetic overflow");
    return a * b;
}

static inline void *pdc_alloc(size_t count, size_t element, MPI_Comm comm)
{
    const size_t bytes = pdc_product(count, element, comm);
    void *ptr = calloc(bytes ? bytes : 1, 1);
    if (!ptr) {
        char message[240];
        snprintf(message, sizeof(message),
                 "cannot allocate %zu bytes: %s; reduce size/ranks per node or request more RAM",
                 bytes, strerror(errno));
        pdc_die(comm, message);
    }
    return ptr;
}

static inline int pdc_count(size_t count, MPI_Comm comm)
{
    if (count > INT_MAX) pdc_die(comm, "MPI count exceeds INT_MAX; reduce the problem size");
    return (int)count;
}

static inline void pdc_block(long n, int p, int rank, long *start, long *count)
{
    const long q = n / p, remainder = n % p;
    *count = q + (rank < remainder);
    *start = rank * q + (rank < remainder ? rank : remainder);
}

static inline long pdc_number(const char *value, long lower, long upper)
{
    char *end = NULL;
    errno = 0;
    long result = strtol(value, &end, 10);
    if (errno || !*value || *end || result < lower || result > upper)
        pdc_die(MPI_COMM_WORLD, "invalid numeric argument (use --help)");
    return result;
}

static inline pdc_options pdc_parse(int argc, char **argv, long default_size)
{
    pdc_options o;
    o.size = default_size; o.threads = omp_get_max_threads(); o.reps = 1;
    o.validate = 1; o.wait_first = 0; o.micro = 0; o.bytes = 1024;
    o.csv = NULL; o.run_id = "manual"; o.experiment = "manual";
    o.schedule = "static"; o.partition = "rows";
    for (int i = 1; i < argc; ++i) {
        const char *key = argv[i];
        if (!strcmp(key, "--help") || !strcmp(key, "-h")) {
            int initialized = 0, rank = 0;
            MPI_Initialized(&initialized);
            if (initialized) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (!rank) printf(
                "Usage: %s [options]\n"
                "  --size N            matrix dimension/users (default %ld)\n"
                "  --threads T         OpenMP threads (default OMP_NUM_THREADS/runtime)\n"
                "  --reps R            measured repetitions; correctness gate is additional\n"
                "  --csv PATH          append raw rank and slowest-rank wall records\n"
                "  --run-id ID         provenance identifier (default manual)\n"
                "  --experiment NAME   experiment label (default manual)\n"
                "  --schedule static|dynamic  Q2 row schedule (dynamic chunk 64)\n"
                "  --partition rows|links     Q2 halo partition override\n"
                "  --wait-first        Q2 halo baseline: wait before computing\n"
                "  --micro             Q3 collective-only benchmark\n"
                "  --bytes B           Q3 micro vector bytes, multiple of sizeof(double)\n"
                "  --validate          correctness checks (always enabled)\n"
                "See README.md for the executable-to-task/version mapping.\n",
                argv[0], default_size);
            if (initialized) MPI_Finalize();
            exit(EXIT_SUCCESS);
        }
        if (!strcmp(key, "--validate")) continue;
        if (!strcmp(key, "--wait-first")) { o.wait_first = 1; continue; }
        if (!strcmp(key, "--micro")) { o.micro = 1; continue; }
        if (i + 1 == argc) pdc_die(MPI_COMM_WORLD, "option requires a value (use --help)");
        const char *v = argv[++i];
        if (!strcmp(key, "--size")) o.size = pdc_number(v, 1, INT_MAX);
        else if (!strcmp(key, "--threads")) o.threads = (int)pdc_number(v, 1, INT_MAX);
        else if (!strcmp(key, "--reps")) o.reps = (int)pdc_number(v, 1, 1000000);
        else if (!strcmp(key, "--bytes")) o.bytes = pdc_number(v, 0, LONG_MAX);
        else if (!strcmp(key, "--csv")) o.csv = v;
        else if (!strcmp(key, "--run-id")) o.run_id = v;
        else if (!strcmp(key, "--experiment")) o.experiment = v;
        else if (!strcmp(key, "--schedule")) o.schedule = v;
        else if (!strcmp(key, "--partition")) o.partition = v;
        else pdc_die(MPI_COMM_WORLD, "unknown option (use --help)");
    }
    if (strcmp(o.schedule, "static") && strcmp(o.schedule, "dynamic"))
        pdc_die(MPI_COMM_WORLD, "--schedule must be static or dynamic");
    if (strcmp(o.partition, "rows") && strcmp(o.partition, "links"))
        pdc_die(MPI_COMM_WORLD, "--partition must be rows or links");
    if (o.threads < 1) pdc_die(MPI_COMM_WORLD, "invalid OpenMP thread count");
    omp_set_dynamic(0);
    omp_set_num_threads(o.threads);
    return o;
}

static inline void pdc_csv_text(FILE *f, const char *s)
{
    fputc('"', f);
    for (; *s; ++s) { if (*s == '"') fputc('"', f); fputc(*s, f); }
    fputc('"', f);
}

static inline void pdc_csv_double(FILE *f, double value)
{
    if (isfinite(value)) fprintf(f, "%.12g", value);
    else fputs("NA", f);
}

/* Called only AFTER a timed region. The wall row is a maximum, never a sum
 * of rank times. Rank records retain overlapping MPI-call and compute samples;
 * wait_s is INCLUDED in comm_s. Collective wire bytes are implementation-specific. */
static inline void pdc_record(MPI_Comm comm, const pdc_options *o,
                              const char *question, const char *version,
                              int repetition, int step, int steps,
                              double local_elapsed, const pdc_metrics *m, int correct)
{
    int rank, p;
    MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &p);
    double wall = 0;
    int all_correct = 0;
    MPI_Reduce(&local_elapsed, &wall, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Allreduce(&correct, &all_correct, 1, MPI_INT, MPI_MIN, comm);
    double values[8] = {local_elapsed, m->compute_s, m->comm_s, m->wait_s,
                       m->stage_s[0], m->stage_s[1], m->stage_s[2], m->stage_s[3]};
    double *all = rank ? NULL : pdc_alloc((size_t)p * 8, sizeof(double), comm);
    long long *bytes = rank ? NULL : pdc_alloc((size_t)p, sizeof(long long), comm);
    char name[MPI_MAX_PROCESSOR_NAME] = {0}; int name_length = 0;
    MPI_Get_processor_name(name, &name_length);
    char *names = rank ? NULL : pdc_alloc((size_t)p, MPI_MAX_PROCESSOR_NAME, comm);
    MPI_Gather(values, 8, MPI_DOUBLE, all, 8, MPI_DOUBLE, 0, comm);
    MPI_Gather(&m->bytes_received, 1, MPI_LONG_LONG_INT, bytes, 1, MPI_LONG_LONG_INT, 0, comm);
    MPI_Gather(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, comm);
    if (!rank) {
        int nodes = 0;
        for (int r = 0; r < p; ++r) {
            int seen = 0;
            for (int k = 0; k < r; ++k)
                if (!strcmp(names + r * MPI_MAX_PROCESSOR_NAME,
                            names + k * MPI_MAX_PROCESSOR_NAME)) seen = 1;
            nodes += !seen;
        }
        printf("%s %s size=%ld p=%d threads=%d nodes=%d rep=%d step=%d wall=%.9g s correctness=%s\n",
               question, version, o->size, p, o->threads, nodes, repetition, step,
               wall, all_correct ? "PASS" : "FAIL");
        if (o->csv) {
            FILE *f = fopen(o->csv, "a+");
            if (!f) pdc_die(comm, "cannot open CSV path; create parent directory and check permissions");
            if (fseek(f, 0, SEEK_END)) pdc_die(comm, "cannot seek CSV");
            if (ftell(f) == 0) fputs(
                "schema_version,timestamp,run_id,experiment,question,version,mode,size,steps,processes,threads,nodes,repetition,step,record,rank,hostname,elapsed_s,compute_s,comm_s,wait_s,bytes_received,correct,stage0_s,stage1_s,stage2_s,stage3_s,vector_bytes,schedule,partition,wait_first\n", f);
            char timestamp[32]; time_t now = time(NULL);
            struct tm *utc = gmtime(&now);
            if (!utc || !strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", utc))
                strcpy(timestamp, "NA");
            for (int r = -1; r < p; ++r) {
                fputs("1,", f); pdc_csv_text(f, timestamp); fputc(',', f);
                pdc_csv_text(f, o->run_id); fputc(',', f);
                pdc_csv_text(f, o->experiment); fputc(',', f);
                pdc_csv_text(f, question); fputc(',', f);
                pdc_csv_text(f, version); fputc(',', f);
                pdc_csv_text(f, o->micro ? "micro" : "workload");
                fprintf(f, ",%ld,%d,%d,%d,%d,%d,%d,%s,%d,", o->size, steps, p,
                        o->threads, nodes, repetition, step, r < 0 ? "wall" : "rank", r);
                pdc_csv_text(f, r < 0 ? "ALL" : names + r * MPI_MAX_PROCESSOR_NAME);
                fputc(',', f); pdc_csv_double(f, r < 0 ? wall : all[r * 8]);
                for (int k = 1; k < 4; ++k) {
                    fputc(',', f); pdc_csv_double(f, r < 0 ? NAN : all[r * 8 + k]);
                }
                if (r < 0 || bytes[r] < 0) fputs(",NA", f);
                else fprintf(f, ",%lld", bytes[r]);
                fprintf(f, ",%s", all_correct ? "PASS" : "FAIL");
                for (int k = 4; k < 8; ++k) {
                    fputc(',', f); pdc_csv_double(f, r < 0 ? NAN : all[r * 8 + k]);
                }
                fprintf(f, ",%ld,", o->micro ? o->bytes : 0L);
                pdc_csv_text(f, o->schedule); fputc(',', f);
                pdc_csv_text(f, o->partition); fprintf(f, ",%d\n", o->wait_first);
            }
            if (ferror(f) || fclose(f)) pdc_die(comm, "CSV write failed (disk full or I/O error)");
        }
        free(all); free(bytes); free(names);
    }
    if (!all_correct) pdc_die(comm, "correctness failed; measurements are invalid");
}

#endif
