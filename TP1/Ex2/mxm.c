
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 400  // Easily changeable for experiments

// Allocate a square matrix (n x n) of doubles
static double** allocate_matrix(int n) {
    double** mat = malloc(n * sizeof(double*));
    if (!mat) {
        fprintf(stderr, "Memory allocation failed for matrix rows\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        mat[i] = malloc(n * sizeof(double));
        if (!mat[i]) {
            fprintf(stderr, "Memory allocation failed for row %d\n", i);
            // Clean up previously allocated rows
            for (int j = 0; j < i; j++) free(mat[j]);
            free(mat);
            exit(EXIT_FAILURE);
        }
    }
    return mat;
}

// Free a square matrix allocated with allocate_matrix
static void free_matrix(double** mat, int n) {
    if (!mat) return;
    for (int i = 0; i < n; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Initialize matrix with random values in [0,1)
static void initialize_matrix(double** mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

// Standard (naive) matrix multiplication: i-j-k order (poor locality)
static void matrix_multiply_standard(int n, const double** a, const double** b, double** c) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
}

// Optimized: i-k-j order + accumulator hoisting (better spatial & temporal locality)
static void matrix_multiply_optimized(int n, const double** a, const double** b, double** c) {
    // Zero-initialize C (separate pass for clarity)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = 0.0;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            const double aik = a[i][k];  // Hoist to reduce loads
            for (int j = 0; j < n; j++) {
                c[i][j] += aik * b[k][j];
            }
        }
    }
}

// Compare two matrices for approximate equality (floating-point safe)
static int matrices_equal(int n, const double** c1, const double** c2) {
    int errors = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double diff = c1[i][j] - c2[i][j];
            if (diff > 1e-10 || diff < -1e-10) {
                errors++;
            }
        }
    }
    return errors;
}

int main(void) {
    const int n = MATRIX_SIZE;
    printf("=== Matrix Multiplication: Loop Order Optimization ===\n");
    printf("Matrix size: %d x %d\n\n", n, n);

    srand(42);  // Fixed seed → reproducible results

    double** a  = allocate_matrix(n);
    double** b  = allocate_matrix(n);
    double** c1 = allocate_matrix(n);  // Standard result
    double** c2 = allocate_matrix(n);  // Optimized result

    initialize_matrix(a, n);
    initialize_matrix(b, n);

    // --- Standard version ---
    clock_t start = clock();
    matrix_multiply_standard(n, (const double**)a, (const double**)b, c1);
    double time_std = (double)(clock() - start) / CLOCKS_PER_SEC;

    // --- Optimized version ---
    start = clock();
    matrix_multiply_optimized(n, (const double**)a, (const double**)b, c2);
    double time_opt = (double)(clock() - start) / CLOCKS_PER_SEC;

    // Performance metrics
    double flops          = 2.0 * n * n * n;
    double gflops_std     = (flops / 1e9) / time_std;
    double gflops_opt     = (flops / 1e9) / time_opt;

    double bytes_moved    = 24.0 * n * n;  // 3 matrices × n² × 8 bytes (rough estimate)
    double bw_std         = (bytes_moved / 1e9) / time_std;   // GB/s
    double bw_opt         = (bytes_moved / 1e9) / time_opt;

    // Output results
    printf("RESULTS:\n");
    printf("--------\n");

    printf("1. STANDARD (i-j-k):\n");
    printf("   Time:       %.4f s\n", time_std);
    printf("   GFLOPS:     %.2f\n", gflops_std);
    printf("   Bandwidth:  %.2f GB/s\n\n", bw_std);

    printf("2. OPTIMIZED (i-k-j):\n");
    printf("   Time:       %.4f s\n", time_opt);
    printf("   GFLOPS:     %.2f\n", gflops_opt);
    printf("   Bandwidth:  %.2f GB/s\n\n", bw_opt);

    printf("3. COMPARISON:\n");
    printf("   Speedup:              %.2f ×\n", time_std / time_opt);
    printf("   Performance gain:     +%.1f%%\n",
           ((gflops_opt - gflops_std) / gflops_std) * 100);
    printf("   Bandwidth gain:       +%.1f%%\n\n",
           ((bw_opt - bw_std) / bw_std) * 100);

    // Verification
    int errors = matrices_equal(n, (const double**)c1, (const double**)c2);
    printf("4. VERIFICATION:\n");
    if (errors == 0) {
        printf("   Results identical (within 1e-10 tolerance)\n");
    } else {
        printf("   %d elements differ (possible floating-point issues)\n", errors);
    }

    // Cleanup
    free_matrix(a,  n);
    free_matrix(b,  n);
    free_matrix(c1, n);
    free_matrix(c2, n);

    return EXIT_SUCCESS;
}