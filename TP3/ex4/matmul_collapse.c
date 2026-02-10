#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef M_SIZE
#define M_SIZE 1000
#endif

#ifndef N_SIZE
#define N_SIZE 1000
#endif

int main(int argc, char *argv[]) {
    int m = M_SIZE;
    int n = N_SIZE;
    int num_threads = 1;
    
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    
    omp_set_num_threads(num_threads);
    
    // Allocate memory dynamically
    double *a = (double *)malloc(m * n * sizeof(double));
    double *b = (double *)malloc(n * m * sizeof(double));
    double *c = (double *)malloc(m * m * sizeof(double));
    
    if (!a || !b || !c) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize matrices
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = (i + 1) + (j + 1);
        }
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            b[i * m + j] = (i + 1) - (j + 1);
        }
    }
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            c[i * m + j] = 0;
        }
    }
    
    // Matrix multiplication with timing
    double start_time = omp_get_wtime();
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < n; k++) {
                c[i * m + j] += a[i * n + k] * b[k * m + j];
            }
        }
    }
    
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    
    // Output: threads,time
    printf("%d,%.6f\n", num_threads, elapsed_time);
    
    // Free memory
    free(a);
    free(b);
    free(c);
    
    return 0;
}
