#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Row-major access macros
#define IDX(i, j, n)   ((i) * (n) + (j))
#define ELEM(mat, i, j, n)  ((mat)[IDX((i), (j), (n))])

// Configuration
#define DEFAULT_N            800
#define DEFAULT_BLOCK_SIZE   64

// Allocate flat n×n matrix
static double* allocate_matrix(int n) {
    double* mat = malloc((size_t)n * n * sizeof(double));
    if (!mat) {
        fprintf(stderr, "Memory allocation failed (%zu bytes)\n",
                (size_t)n * n * sizeof(double));
        exit(EXIT_FAILURE);
    }
    return mat;
}

static void free_matrix(double* mat) {
    free(mat);
}

// Fill matrix with random values in [0,1)
static void initialize_matrix(double* mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ELEM(mat, i, j, n) = (double)rand() / RAND_MAX;
        }
    }
}

// Blocked matrix multiplication (triple tiling)
static void matrix_multiply_blocked(int n, int bs,
                                    const double* restrict a,
                                    const double* restrict b,
                                    double* restrict c) {
    memset(c, 0, (size_t)n * n * sizeof(double));

    for (int ii = 0; ii < n; ii += bs) {
        int i_end = (ii + bs < n) ? ii + bs : n;
        for (int jj = 0; jj < n; jj += bs) {
            int j_end = (jj + bs < n) ? jj + bs : n;
            for (int kk = 0; kk < n; kk += bs) {
                int k_end = (kk + bs < n) ? kk + bs : n;

                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        double aik = ELEM(a, i, k, n);
                        for (int j = jj; j < j_end; j++) {
                            ELEM(c, i, j, n) += aik * ELEM(b, k, j, n);
                        }
                    }
                }
            }
        }
    }
}

// Reference: optimized non-blocked (i-k-j order)
static void matrix_multiply_ref(int n,
                                const double* restrict a,
                                const double* restrict b,
                                double* restrict c) {
    memset(c, 0, (size_t)n * n * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double aik = ELEM(a, i, k, n);
            for (int j = 0; j < n; j++) {
                ELEM(c, i, j, n) += aik * ELEM(b, k, j, n);
            }
        }
    }
}

// Count elements differing by more than tolerance
static int count_differences(int n, const double* c1, const double* c2) {
    int errors = 0;
    const double tol = 1e-9;
    for (int i = 0; i < n * n; i++) {
        double diff = c1[i] - c2[i];
        if (diff > tol || diff < -tol) errors++;
    }
    return errors;
}

int main(int argc, char *argv[]) {
    int n = (argc >= 2) ? atoi(argv[1]) : DEFAULT_N;
    int bs_def = (argc >= 3) ? atoi(argv[2]) : DEFAULT_BLOCK_SIZE;

    if (n <= 0 || bs_def <= 0) {
        fprintf(stderr, "Usage: %s [n] [block_size]\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(42);  // Reproducible results

    printf("TP1 - Ex3: Blocked Matrix Multiplication\n");
    printf("Matrix size: %d × %d\n\n", n, n);

    double *a = allocate_matrix(n);
    double *b = allocate_matrix(n);
    double *c_block = allocate_matrix(n);
    double *c_ref   = allocate_matrix(n);

    initialize_matrix(a, n);
    initialize_matrix(b, n);

    // Reference timing
    clock_t t_start = clock();
    matrix_multiply_ref(n, a, b, c_ref);
    double t_ref = (double)(clock() - t_start) / CLOCKS_PER_SEC;

    printf("Reference (i-k-j non-blocked): %.4f s\n\n", t_ref);

    // Block sizes to evaluate
    int block_sizes[] = {8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 192, 256, 384, 512, n};
    int n_tests = sizeof(block_sizes) / sizeof(block_sizes[0]);

    printf("Block Size | Time (s) | GFLOPS | BW (GB/s) | Speedup vs Ref\n");
    printf("-----------|----------|--------|-----------|----------------\n");

    double best_time = 1e99;
    int best_bs = -1;

    for (int i = 0; i < n_tests; i++) {
        int bs = block_sizes[i];

        t_start = clock();
        matrix_multiply_blocked(n, bs, a, b, c_block);
        double t = (double)(clock() - t_start) / CLOCKS_PER_SEC;

        double flops  = 2.0 * (double)n * n * n;
        double gflops = (flops / 1e9) / t;
        double bytes  = 24.0 * (double)n * n;          // rough total movement
        double bw     = (bytes / 1e9) / t;
        double speedup = t_ref / t;

        printf("%10d | %8.4f | %6.2f | %9.2f | %13.2f×\n",
               bs, t, gflops, bw, speedup);

        if (t < best_time) {
            best_time = t;
            best_bs   = bs;
        }
    }

    printf("\nOptimal block size: %d\n", best_bs);
    printf("Justification:\n");
    printf("  - Maximizes temporal locality (blocks fit in L1/L2 cache)\n");
    printf("  - Good balance: enough work to hide overhead, avoids thrashing\n");
    printf("  - Aligns with lecture principles: high reuse while data is cached\n");

    // Optional correctness check (useful for small n)
    if (n <= 512) {
        int errs = count_differences(n, c_block, c_ref);
        printf("\nVerification (last block size): %d differences (tol=1e-9)\n", errs);
    }

    free_matrix(a);
    free_matrix(b);
    free_matrix(c_block);
    free_matrix(c_ref);

    return EXIT_SUCCESS;
}