#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000

void init_matrix(int n, double *A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i*n + j] = (double)(i + j);
        }
    }
}

void print_matrix(int n, double *A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%6.1f ", A[i*n + j]);
        }
        printf("\n");
    }
}

int main() {
    double *A;
    double sum = 0.0;
    double start_seq, end_seq, start_par, end_par;

    A = (double*) malloc(N * N * sizeof(double));
    if (A == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    start_par = omp_get_wtime();

    #pragma omp parallel
{
    #pragma omp master
    {
        printf("Master thread %d initializing matrix...\n", omp_get_thread_num());
        printf("Total threads: %d\n", omp_get_num_threads());
        init_matrix(N, A);
    }

    #pragma omp single
    {
        printf("Single thread %d printing (skipped for large N)...\n", omp_get_thread_num());
    }

    #pragma omp for reduction(+:sum)
    for (int i = 0; i < N*N; i++) {
        sum += A[i];
    }
}

    end_par = omp_get_wtime();
    printf("Sum = %lf\n", sum);
    printf("Execution time = %lf seconds\n", end_par - start_par);

    free(A);
    return 0;
}