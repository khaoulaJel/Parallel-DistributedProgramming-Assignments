#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#ifndef VAL_N
#define VAL_N 500
#endif
#ifndef VAL_D
#define VAL_D 100
#endif

void random_number(double* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (double)rand() / (double)(RAND_MAX - 1);
    }
}

int main(int argc, char *argv[]) {
    int n = VAL_N, diag = VAL_D;
    int i, j, iteration = 0;
    double norme;
    int num_threads = 1;

    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    
    omp_set_num_threads(num_threads);

    double *a = (double*)malloc(n * n * sizeof(double));
    double *x = (double*)malloc(n * sizeof(double));
    double *x_courant = (double*)malloc(n * sizeof(double));
    double *b = (double*)malloc(n * sizeof(double));

    if (!a || !x || !x_courant || !b) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    double t_cpu_0, t_cpu_1, t_cpu;

    srand(421);
    random_number(a, n * n);
    random_number(b, n);

    // Make matrix diagonally dominant
    for (i = 0; i < n; i++) {
        a[i * n + i] += diag;
    }

    // Initialize x
    for (i = 0; i < n; i++) {
        x[i] = 1.0;
    }

    t_cpu_0 = omp_get_wtime();

    while (1) {
        iteration++;

        // Parallelized Jacobi iteration
        #pragma omp parallel for private(j) schedule(static)
        for (i = 0; i < n; i++) {
            double sum = 0.0;
            for (j = 0; j < n; j++) {
                if (j != i) {
                    sum += a[i * n + j] * x[j];
                }
            }
            x_courant[i] = (b[i] - sum) / a[i * n + i];
        }

        // Calculate convergence criterion
        double absmax = 0.0;
        #pragma omp parallel for reduction(max:absmax)
        for (i = 0; i < n; i++) {
            double curr = fabs(x[i] - x_courant[i]);
            if (curr > absmax)
                absmax = curr;
        }
        norme = absmax / n;

        if ((norme <= DBL_EPSILON) || (iteration >= n)) break;

        // Copy new values
        #pragma omp parallel for
        for (i = 0; i < n; i++) {
            x[i] = x_courant[i];
        }
    }

    t_cpu_1 = omp_get_wtime();
    t_cpu = t_cpu_1 - t_cpu_0;

    printf("%d,%.6f,%d,%.3E\n", num_threads, t_cpu, iteration, norme);

    free(a); free(x); free(x_courant); free(b);
    return EXIT_SUCCESS;
}
