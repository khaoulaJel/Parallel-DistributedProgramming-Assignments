#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size_mpi;
    long long N;
    double serial_time, parallel_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size_mpi);

    if (argc != 2) {
        if (rank == 0)
            printf("Usage: %s <N>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    N = atoll(argv[1]);
    if (N <= 0) {
        if (rank == 0)
            printf("N must be positive.\n");
        MPI_Finalize();
        return 1;
    }

    // Serial computation
    double pi_serial = 0.0;
    if (rank == 0) {
        double start = MPI_Wtime();
        double sum = 0.0;
        for (long long i = 0; i < N; i++) {
            double x = (i + 0.5) / N;
            sum += 1.0 / (1.0 + x * x);
        }
        pi_serial = 4.0 * sum / N;
        serial_time = MPI_Wtime() - start;
        printf("Serial Pi       : %.15f\n", pi_serial);
        printf("Reference Pi    : %.15f\n", M_PI);
        printf("Serial error    : %.3e\n", fabs(pi_serial - M_PI));
        printf("Serial time     : %f seconds\n\n", serial_time);
    }

    // Broadcast serial_time so all ranks have it for speedup calc
    MPI_Bcast(&serial_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Parallel computation 
    MPI_Barrier(MPI_COMM_WORLD);
    double par_start = MPI_Wtime();

    // Distribute N iterations across processes 
    long long base  = N / size_mpi;
    long long extra = N % size_mpi;

    // rank < extra gets one extra iteration
    long long local_start = rank * base + (rank < extra ? rank : extra);
    long long local_end   = local_start + base + (rank < extra ? 1 : 0);

    double local_sum = 0.0;
    for (long long i = local_start; i < local_end; i++) {
        double x = (i + 0.5) / N;
        local_sum += 1.0 / (1.0 + x * x);
    }

    // Reduce all local sums to rank 0
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    parallel_time = MPI_Wtime() - par_start;

    if (rank == 0) {
        double pi_parallel = 4.0 * global_sum / N;
        double speedup    = serial_time / parallel_time;
        double efficiency = speedup / size_mpi * 100.0;

        printf("Parallel Pi     : %.15f\n", pi_parallel);
        printf("Parallel error  : %.3e\n", fabs(pi_parallel - M_PI));
        printf("Parallel time (%d procs): %f seconds\n", size_mpi, parallel_time);
        printf("Speedup         : %f\n", speedup);
        printf("Efficiency      : %f%%\n\n", efficiency);
    }

    MPI_Finalize();
    return 0;
}