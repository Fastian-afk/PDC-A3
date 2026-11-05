/* Imaad Fazal | 23i-0656 | Section C | Assignment 3
 * Q1 shared implementation: dense matrix multiplication and MPI row rings.
 * All MPI calls execute on the main thread outside OpenMP regions.
 */
#ifndef PDC_Q1_COMMON_H
#define PDC_Q1_COMMON_H
#include "../common/pdc.h"
#include <math.h>
#include <limits.h>
#include <string.h>

#ifndef Q1_VARIANT
#error "Define Q1_VARIANT from 1 (serial ijk) through 6 (nonblocking ring)."
#endif

static const char *q1_version(void)
{
    static const char *names[] = {"", "serial-ijk", "serial-ikj", "omp", "A", "B", "C"};
    return names[Q1_VARIANT];
}

static double q1_a(long i, long j) { return (double)((2 * i + j) % 7 - 3); }
static double q1_b(long i, long j) { return (double)((i + 3 * j) % 5 - 1); }

/* i-k-j walks contiguous B and C elements. Each OpenMP worker owns whole
 * output rows, so no atomics or reductions are needed in this kernel. */
static void q1_multiply(const double *a, const double *b, double *c,
                        long local_rows, long n, long k_start, long k_rows,
                        int threads)
{
    if (Q1_VARIANT == 1) {
        for (long i = 0; i < local_rows; ++i)
            for (long j = 0; j < n; ++j) {
                double sum = 0.0;
                for (long k = 0; k < k_rows; ++k)
                    sum += a[(size_t)i * (size_t)n + (size_t)(k_start + k)] *
                           b[(size_t)k * (size_t)n + (size_t)j];
                c[(size_t)i * (size_t)n + (size_t)j] += sum;
            }
    } else {
        #pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
        for (long i = 0; i < local_rows; ++i) {
            double *out = c + (size_t)i * (size_t)n;
            const double *arow = a + (size_t)i * (size_t)n;
            for (long k = 0; k < k_rows; ++k) {
                const double av = arow[k_start + k];
                const double *brow = b + (size_t)k * (size_t)n;
                for (long j = 0; j < n; ++j)
                    out[j] += av * brow[j];
            }
        }
    }
}

/* Direct scalar oracle is independent of storage/distribution and kernel order. */
static double q1_reference(long i, long j, long n)
{
    double sum = 0.0;
    for (long k = 0; k < n; ++k) sum += q1_a(i, k) * q1_b(k, j);
    return sum;
}

static int q1_check(const double *c, long n, long start, long local_rows,
                     int rank, int print_values, MPI_Comm comm)
{
    const long sr[5] = {0, 0, n > 1 ? 1 : 0, n / 2, n - 1};
    const long sc[5] = {0, n > 1 ? 1 : 0, 0, n / 2, n - 1};
    static const double expected_large[3][5] = {
        {2, -2, 7, 3, 9}, {5, -10, 15, -12, -5}, {-7, 5, 20, 9, 17}
    };
    static const double expected8[64] = {
        0,6,-8,-2,-11,0,6,-8, 17,-11,1,-2,-10,17,-11,1,
        -8,0,3,-9,19,-8,0,3, -5,4,-2,12,6,-5,4,-2,
        5,-6,-7,12,-14,5,-6,-7, 1,-9,16,-9,1,1,-9,16,
        -10,16,-3,-2,9,-10,16,-3, 0,6,-8,-2,-11,0,6,-8
    };
    int ok = 1;
    /* A repeats every 7 rows and B every 5 columns. Build 35 scalar-dot
     * reference values, then check EVERY local result even for large runs.
     * Small non-table runs use a direct dot for each output as a second path. */
    double tile[7][5];
    if (n > 64)
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 5; ++j) tile[i][j] = q1_reference(i,j,n);
    for (long li = 0; li < local_rows; ++li)
        for (long j = 0; j < n; ++j) {
            const long i = start + li;
            const double expected = n == 8 ? expected8[i * n + j] :
                                    n <= 64 ? q1_reference(i,j,n) : tile[i % 7][j % 5];
            const double got = c[(size_t)li * (size_t)n + (size_t)j];
            if (got != expected) {
                if (ok) fprintf(stderr,"Q1 %s rank %d: C(%ld,%ld)=%.17g expected %.17g\n",
                                q1_version(),rank,i,j,got,expected);
                ok = 0;
            }
        }
    for (int k = 0; k < 5; ++k) {
        if (sr[k] < start || sr[k] >= start + local_rows) continue;
        const double got = c[(size_t)(sr[k]-start)*(size_t)n+(size_t)sc[k]];
        const int table = n == 1000 ? 0 : n == 2000 ? 1 : n == 4000 ? 2 : -1;
        const double expected = table >= 0 ? expected_large[table][k] : q1_reference(sr[k],sc[k],n);
        if (got != expected) {
            fprintf(stderr,"Q1 %s rank %d: sample C(%ld,%ld)=%.17g expected %.17g\n",
                    q1_version(),rank,sr[k],sc[k],got,expected);
            ok = 0;
        }
        if (print_values && n != 8) {
            printf("Q1 version=%s rank=%d N=%ld C(%ld,%ld)=%.0f expected=%.0f\n",
                   q1_version(),rank,n,sr[k],sc[k],got,expected);
        }
    }
    int global_ok;
    MPI_Allreduce(&ok,&global_ok,1,MPI_INT,MPI_MIN,comm);
    return global_ok;
}

static void q1_small_print(const double *c, long n, int rank, int p,
                           const int *counts, const int *displs, MPI_Comm comm)
{
    if (n != 8) return;
    double *all = NULL;
    if (rank == 0) all = pdc_alloc(64,sizeof(double),comm);
    MPI_Gatherv(c,counts[rank],MPI_DOUBLE,all,counts,displs,MPI_DOUBLE,0,comm);
    if (rank == 0) {
        printf("Q1 version=%s N=8 p=%d complete matrix C (Table 1a):\n",q1_version(),p);
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) printf("%5.0f%s",all[i*8+j],j==7?"\n":" ");
        }
        free(all);
    }
}

static int q1_main(int argc, char **argv)
{
    MPI_Init(&argc,&argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, p;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&p);
    pdc_options opt = pdc_parse(argc,argv,8);
    const long n = opt.size;
    if (n < 1 || n > INT_MAX) pdc_die(comm,"Q1 --size must be in [1, INT_MAX].");
    if (Q1_VARIANT <= 3 && p != 1)
        pdc_die(comm,"Serial/OpenMP Q1 versions require one MPI process (use -np 1).");
    /* Serial programs always use one thread, even if OMP_NUM_THREADS is set. */
    if (Q1_VARIANT <= 2) opt.threads = 1;
    omp_set_dynamic(0);
    /* Probe outside the timed region: OMP_THREAD_LIMIT may cap the requested
     * team. CSV metadata records the actual team used by the kernels. */
    if (Q1_VARIANT >= 3) {
        int actual_threads = 1;
        #pragma omp parallel num_threads(opt.threads)
        {
            #pragma omp single
            actual_threads = omp_get_num_threads();
        }
        int minimum_threads, maximum_threads;
        MPI_Allreduce(&actual_threads,&minimum_threads,1,MPI_INT,MPI_MIN,comm);
        MPI_Allreduce(&actual_threads,&maximum_threads,1,MPI_INT,MPI_MAX,comm);
        if (minimum_threads != maximum_threads)
            pdc_die(comm,"Q1 actual OpenMP team sizes differ across ranks; check OMP_THREAD_LIMIT and launcher environment.");
        opt.threads = actual_threads;
    }
    const int threads = opt.threads;
    omp_set_num_threads(threads);
    long start, rows;
    pdc_block(n,p,rank,&start,&rows);
    long max_rows = n / p + (n % p != 0);
    if ((size_t)n > SIZE_MAX / (size_t)n) pdc_die(comm,"Q1 matrix dimension overflows size_t.");
    const size_t total = (size_t)n * (size_t)n;
    const size_t local_n = (size_t)rows * (size_t)n;
    const size_t block_n = (size_t)max_rows * (size_t)n;
    pdc_count(local_n,comm);
    pdc_count(block_n,comm);
    /* Scatterv/Bcast/Gatherv have int count and displacement arguments. */
    if (Q1_VARIANT <= 4) pdc_count(total,comm);
    int *counts = pdc_alloc((size_t)p,sizeof(int),comm);
    int *displs = pdc_alloc((size_t)p,sizeof(int),comm);
    for (int r=0;r<p;++r) {
        long rs, rn;
        pdc_block(n,p,r,&rs,&rn);
        counts[r] = pdc_count((size_t)rn*(size_t)n,comm);
        /* Large ring runs do not use Gatherv displacements. */
        displs[r] = (Q1_VARIANT <= 4 || n == 8) ?
                    pdc_count((size_t)rs*(size_t)n,comm) : 0;
    }
    double *a = pdc_alloc(local_n,sizeof(double),comm);
    double *c = pdc_alloc(local_n,sizeof(double),comm);
    double *root_a = NULL, *root_c = NULL;
    double *b[2] = {NULL,NULL};
    b[0] = pdc_alloc(Q1_VARIANT <= 4 ? total : block_n,sizeof(double),comm);
    if (Q1_VARIANT >= 5) b[1] = pdc_alloc(block_n,sizeof(double),comm);
    if (Q1_VARIANT == 4 && rank == 0) {
        root_a = pdc_alloc(total,sizeof(double),comm);
        root_c = pdc_alloc(total,sizeof(double),comm);
        for (long i=0;i<n;++i)
            for (long j=0;j<n;++j) {
                root_a[(size_t)i*(size_t)n+(size_t)j] = q1_a(i,j);
                b[0][(size_t)i*(size_t)n+(size_t)j] = q1_b(i,j);
            }
    } else if (Q1_VARIANT != 4) {
        for (long i=0;i<rows;++i)
            for (long j=0;j<n;++j) a[(size_t)i*(size_t)n+(size_t)j]=q1_a(start+i,j);
    }
    if (rank == 0) {
        const size_t b_elems = Q1_VARIANT <= 4 ? total : 2 * block_n;
        printf("Q1 version=%s N=%ld processes=%d threads=%d repetitions=%d\n",
               q1_version(),n,p,threads,opt.reps);
        printf("rank0 matrix-buffer bytes=%zu (includes root-only A/C in Version A); "
               "initialization and verification excluded from timing\n",
               (local_n*2+b_elems+(Q1_VARIANT==4?2*total:0))*sizeof(double));
    }
    /* rep=-1 is the mandatory untimed correctness gate. Every timed repetition
     * starts from an identical initialized state and is checked afterwards. */
    for (int rep=-1;rep<opt.reps;++rep) {
        memset(c,0,local_n*sizeof(double));
        if (Q1_VARIANT != 4) {
            const long br = Q1_VARIANT <= 3 ? n : rows;
            const long bs = Q1_VARIANT <= 3 ? 0 : start;
            for (long i=0;i<br;++i)
                for (long j=0;j<n;++j) b[0][(size_t)i*(size_t)n+(size_t)j]=q1_b(bs+i,j);
        }
        pdc_metrics m = {0};
        for (int k=0;k<4;++k) m.stage_s[k]=NAN;
        MPI_Barrier(comm);
        const double t0 = MPI_Wtime();
        if (Q1_VARIANT == 4) {
            double t=MPI_Wtime();
            /* COLLECTIVE communication: rank 0 distributes complete A rows. */
            MPI_Scatterv(root_a,counts,displs,MPI_DOUBLE,a,counts[rank],MPI_DOUBLE,0,comm);
            m.stage_s[0]=MPI_Wtime()-t;
            t=MPI_Wtime();
            MPI_Bcast(b[0],pdc_count(total,comm),MPI_DOUBLE,0,comm);
            m.stage_s[1]=MPI_Wtime()-t;
            t=MPI_Wtime();
            q1_multiply(a,b[0],c,rows,n,0,n,threads);
            m.compute_s=MPI_Wtime()-t;
            m.stage_s[2]=m.compute_s;
            t=MPI_Wtime();
            MPI_Gatherv(c,counts[rank],MPI_DOUBLE,root_c,counts,displs,MPI_DOUBLE,0,comm);
            m.stage_s[3]=MPI_Wtime()-t;
            m.comm_s=m.stage_s[0]+m.stage_s[1]+m.stage_s[3];
            m.bytes_received=rank==0?(long long)(total-local_n)*(long long)sizeof(double):
                            (long long)(local_n+total)*(long long)sizeof(double);
        } else if (Q1_VARIANT <= 3) {
            const double t=MPI_Wtime();
            q1_multiply(a,b[0],c,rows,n,0,n,threads);
            m.compute_s=MPI_Wtime()-t;
        } else {
            const int left=(rank-1+p)%p, right=(rank+1)%p;
            int cur=0;
            for (int step=0;step<p;++step) {
                const int q=(rank+step)%p, next=(rank+step+1)%p;
                long qs, qn, ns, nn;
                pdc_block(n,p,q,&qs,&qn);
                pdc_block(n,p,next,&ns,&nn);
                (void)ns;
                MPI_Request req[2];
                if (Q1_VARIANT == 6 && step < p-1) {
                    const double t=MPI_Wtime();
                    /* POINT-TO-POINT OVERLAP: receive into the alternate buffer;
                     * the current send buffer is read-only through Waitall. */
                    MPI_Irecv(b[1-cur],counts[next],MPI_DOUBLE,right,101,comm,&req[0]);
                    MPI_Isend(b[cur],counts[q],MPI_DOUBLE,left,101,comm,&req[1]);
                    m.comm_s += MPI_Wtime()-t;
                }
                const double tc=MPI_Wtime();
                q1_multiply(a,b[cur],c,rows,n,qs,qn,threads);
                m.compute_s += MPI_Wtime()-tc;
                if (step < p-1) {
                    const double t=MPI_Wtime();
                    if (Q1_VARIANT == 5) {
                        /* BLOCKING ring. Even ranks send first; odd ranks receive
                         * first. For odd p there is still a matching odd receiver
                         * to break the send-only cycle. p=1 never enters here. */
                        if (rank%2 == 0) {
                            MPI_Send(b[cur],counts[q],MPI_DOUBLE,left,101,comm);
                            MPI_Recv(b[1-cur],counts[next],MPI_DOUBLE,right,101,comm,MPI_STATUS_IGNORE);
                        } else {
                            MPI_Recv(b[1-cur],counts[next],MPI_DOUBLE,right,101,comm,MPI_STATUS_IGNORE);
                            MPI_Send(b[cur],counts[q],MPI_DOUBLE,left,101,comm);
                        }
                    } else {
                        MPI_Waitall(2,req,MPI_STATUSES_IGNORE);
                        m.wait_s += MPI_Wtime()-t;
                    }
                    m.comm_s += MPI_Wtime()-t;
                    m.bytes_received += (long long)nn*(long long)n*(long long)sizeof(double);
                    cur=1-cur;
                }
            }
        }
        MPI_Barrier(comm);
        const double elapsed=MPI_Wtime()-t0;
        const int correct=q1_check(c,n,start,rows,rank,rep==-1,comm);
        if (rep == -1) {
            q1_small_print(c,n,rank,p,counts,displs,comm);
            if (rank==0) printf("Q1 version=%s untimed correctness gate: %s\n",q1_version(),correct?"PASS":"FAIL");
        } else {
            pdc_record(comm,&opt,"Q1",q1_version(),rep,-1,1,elapsed,&m,correct);
        }
        if (!correct) pdc_die(comm,"Q1 reference validation failed.");
    }
    free(a); free(c); free(root_a); free(root_c); free(b[0]); free(b[1]);
    free(counts); free(displs);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
#endif
