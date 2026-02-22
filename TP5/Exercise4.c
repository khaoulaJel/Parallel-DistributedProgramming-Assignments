#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void matrixVectorMult(double* A, double* b, double* x, int size) {
    for (int i = 0; i < size; ++i) {
        x[i] = 0.0;
        for (int j = 0; j < size; ++j) {
            x[i] += A[i * size + j] * b[j];
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, size_mpi;
    double *A, *b, *x_serial, *x_parallel;
    int matrix_size;
    double start_time, end_time;
    double serial_time = 0.0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size_mpi);
    
    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <matrix_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    matrix_size = atoi(argv[1]);
    
    if (matrix_size <= 0) {
        if (rank == 0) {
            printf("Matrix size must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // Only process 0 allocates and initializes data
    if (rank == 0) {
        A = malloc(matrix_size * matrix_size * sizeof(double));
        b = malloc(matrix_size * sizeof(double));
        x_serial = malloc(matrix_size * sizeof(double));
        x_parallel = malloc(matrix_size * sizeof(double));
        
        if (!A || !b || !x_serial || !x_parallel) {
            printf("Memory allocation failed.\n");
            MPI_Finalize();
            return 1;
        }
        
        srand48(42);
        
        // Fill A[0][:100] with random values
        int limit = (matrix_size < 100) ? matrix_size : 100;
        for (int j = 0; j < limit; ++j)
            A[0 * matrix_size + j] = drand48();
        
        // Copy A[0][:100] into A[1][100:200] if possible
        if (matrix_size > 1 && matrix_size > 100) {
            int copy_len = (matrix_size - 100 < 100) ? (matrix_size - 100) : 100;
            for (int j = 0; j < copy_len; ++j)
                A[1 * matrix_size + (100 + j)] = A[0 * matrix_size + j];
        }
        
        // Set diagonal
        for (int i = 0; i < matrix_size; ++i)
            A[i * matrix_size + i] = drand48();
        
        // Fill vector b
        for (int i = 0; i < matrix_size; ++i)
            b[i] = drand48();
        
        // Compute serial version for comparison
        start_time = MPI_Wtime();
        matrixVectorMult(A, b, x_serial, matrix_size);
        end_time = MPI_Wtime();
        serial_time = end_time - start_time;
        
        printf("Serial computation time: %f seconds\n", serial_time);
    } else {
        // Other processes allocate only what they need
        A = malloc(matrix_size * matrix_size * sizeof(double));
        b = malloc(matrix_size * sizeof(double));
        x_parallel = malloc(matrix_size * sizeof(double));
        x_serial = NULL;
    }
    
    // Broadcast matrix A and vector b to all processes
    MPI_Bcast(A, matrix_size * matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Synchronize before timing parallel computation
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Calculate rows per process (with remainder handling)
    int rows_per_process = matrix_size / size_mpi;
    int remainder = matrix_size % size_mpi;
    
    // Determine local number of rows
    int local_rows;
    int start_row;
    
    if (rank < remainder) {
        local_rows = rows_per_process + 1;
        start_row = rank * (rows_per_process + 1);
    } else {
        local_rows = rows_per_process;
        start_row = remainder * (rows_per_process + 1) + (rank - remainder) * rows_per_process;
    }
    
    // Allocate local result vector
    double* local_x = malloc(local_rows * sizeof(double));
    
    // Compute local matrix-vector product
    for (int i = 0; i < local_rows; ++i) {
        local_x[i] = 0.0;
        int global_row = start_row + i;
        for (int j = 0; j < matrix_size; ++j) {
            local_x[i] += A[global_row * matrix_size + j] * b[j];
        }
    }
    
    // Gather results from all processes to process 0
    int* recvcounts = NULL;
    int* displs = NULL;
    
    if (rank == 0) {
        recvcounts = malloc(size_mpi * sizeof(int));
        displs = malloc(size_mpi * sizeof(int));
        
        // Calculate receive counts and displacements
        for (int i = 0; i < size_mpi; ++i) {
            if (i < remainder) {
                recvcounts[i] = rows_per_process + 1;
            } else {
                recvcounts[i] = rows_per_process;
            }
            
            if (i == 0) {
                displs[i] = 0;
            } else {
                displs[i] = displs[i - 1] + recvcounts[i - 1];
            }
        }
    }
    
    // Gather all results
    MPI_Gatherv(local_x, local_rows, MPI_DOUBLE, x_parallel, recvcounts, displs, 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Synchronize and get parallel time
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    double parallel_time = end_time - start_time;
    
    // Verify results and print timing
    if (rank == 0) {
        printf("Parallel computation time (%d processes): %f seconds\n", size_mpi, parallel_time);
        printf("Speedup: %f\n", serial_time / parallel_time);
        printf("Efficiency: %f%%\n", (serial_time / parallel_time) / size_mpi * 100);
        
        // Compare both results
        double max_error = 0.0;
        for (int i = 0; i < matrix_size; ++i) {
            double diff = fabs(x_parallel[i] - x_serial[i]);
            if (diff > max_error)
                max_error = diff;
        }
        printf("Maximum difference between Parallel and serial result: %e\n\n", max_error);
    }
    
    // Cleanup
    free(A);
    free(b);
    free(local_x);
    free(x_parallel);
    
    if (rank == 0) {
        free(x_serial);
        free(recvcounts);
        free(displs);
    }
    
    MPI_Finalize();
    return 0;
}