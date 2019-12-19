/* Include benchmark-specific header. */
#include "2mm.h"

double bench_t_start, bench_t_end;

static
double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
        printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start()
{
    bench_t_start = rtclock ();
}

void bench_timer_stop()
{
    bench_t_end = rtclock ();
}

void bench_timer_print()
{
    printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}

static
void init_array(int ni, int nj, int nk, int nl,
                double *alpha,
                double *beta,
                double A[ ni][nk],
                double B[ nk][nj],
                double C[ nj][nl],
                double D[ ni][nl],
                double tmp[ ni][nj])
{
    int i, j;

    *alpha = 1.5;
    *beta = 1.2;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nk; j++)
            A[i][j] = (double) ((i*j+1) % ni) / ni;
    for (i = 0; i < nk; i++)
        for (j = 0; j < nj; j++)
            B[i][j] = (double) (i*(j+1) % nj) / nj;
    for (i = 0; i < nj; i++)
        for (j = 0; j < nl; j++)
            C[i][j] = (double) ((i*(j+3)+1) % nl) / nl;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++)
            D[i][j] = (double) (i*(j+2) % nk) / nk;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++)
            tmp[i][j] = 0.0;
}

static
void print_array(int ni, int nl,
                 double D[ ni][nl]) {
    int i, j;
    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stderr, "begin dump: %s\n", "D");
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
            fprintf(stderr, "%0.2lf ", D[i][j]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\nend   dump: %s\n", "D");
    fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static
void kernel_2mm(int ni, int nj, int nk, int nl,
                double alpha,
                double beta,
                int rank,
                int max_row,
                double tmp[ ni][nj],
                double A[ ni][nk],
                double B[ nk][nj],
                double C[ nj][nl],
                double D[ ni][nl])
{
    int i, j, k;
    for (i = rank * max_row; i < max_row + rank * max_row; i ++) {
        for (j = 0; j < nj; j++) {
            tmp[i][j] = 0.0;
            for (k = 0; k < nk; ++k) {
                tmp[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }
    MPI_Gather(&tmp[rank*max_row], max_row*nj, MPI_DOUBLE, tmp, max_row*nj, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    for (i = rank * max_row; i < max_row + rank * max_row; i ++) {
        for (j = 0; j < nl; j++) {
            D[i][j] *= beta;
            for (k = 0; k < nj; ++k)
                D[i][j] += tmp[i][k] * C[k][j];
        }
    }
    MPI_Gather(&D[rank*max_row], max_row*nl, MPI_DOUBLE, D, max_row*nl, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char** argv)
{

    int err = MPI_Init(&argc, &argv);

    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "Error while starting! \n");
        MPI_Abort(MPI_COMM_WORLD, err);
    }

    int size, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (!rank) {
        printf("Number of threads: %d\n", size);
    }

    int nis[5] = {16, 40, 180, 800, 1600};
    int njs[5] = {18, 50, 190, 900, 1800};
    int nks[5] = {22, 70, 210, 1100, 2200};
    int nls[5] = {24, 80, 220, 1200, 2400};
    char *names[5] = {"MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE"};


    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < 5; i++) {
        ni = nis[i];
        nk = nks[i];
        nj = njs[i];
        nl = nls[i];

        double alpha;
        double beta;
        double (*tmp)[ni][nj] = NULL;
        double (*A)[ni][nk] = NULL;
        double (*B)[nk][nj] = NULL;
        double (*C)[nj][nl] = NULL;
        double (*D)[ni][nl] = NULL;
        tmp = (double (*)[ni][nj]) malloc((ni) * (nj) * sizeof(double));
        A = (double (*)[ni][nk]) malloc((ni) * (nk) * sizeof(double));
        B = (double (*)[nk][nj]) malloc((nk) * (nj) * sizeof(double));
        C = (double (*)[nj][nl]) malloc((nj) * (nl) * sizeof(double));
        D = (double (*)[ni][nl]) malloc((ni) * (nl) * sizeof(double));

        init_array(ni, nj, nk, nl, &alpha, &beta,
                   *A,
                   *B,
                   *C,
                   *D,
                   *tmp);
        double start = MPI_Wtime();
        int max_row = ni/size;
        kernel_2mm(ni, nj, nk, nl,
                   alpha, beta,
                   rank, max_row,
                   *tmp,
                   *A,
                   *B,
                   *C,
                   *D);

        double end = MPI_Wtime();
        if (rank == 0){
            printf("Dataset %s:\nTime = %fs\n", names[i], end-start);
            fflush(stdin);
        }

        if (argc > 42 && !strcmp(argv[0], "")) print_array(ni, nl, *D);
        MPI_Barrier(MPI_COMM_WORLD);
        if (tmp != NULL)
            free((void *) tmp);
        tmp = NULL;
        if (A != NULL)
            free((void *) A);
        A = NULL;
        if (B != NULL)
            free((void *) B);
        B = NULL;
        if (C != NULL)
            free((void *) C);
        C = NULL;
        if (D != NULL)
            free((void *) D);
        D = NULL;
    }



    MPI_Finalize();

    return 0;
}
