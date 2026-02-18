#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void dmvm_v1(int n, int m, double *lhs, double *rhs, double *mat) {
    // Standard parallel loop with an implicit barrier at the end
    #pragma omp parallel for reduction(+:lhs[:m])
    for (int c = 0; c < n; ++c) {
        int offset = m * c;
        for (int r = 0; r < m; ++r) {
            lhs[r] += mat[r + offset] * rhs[c];
        }
    } 
}

int main() {
    const int n = 40000; // columns
    const int m = 600;   // rows
    double *mat = malloc(n * m * sizeof(double));
    double *rhs = malloc(n * sizeof(double));
    double *lhs = malloc(m * sizeof(double));

    for (int c = 0; c < n; ++c) {
        rhs[c] = 1.0;
        for (int r = 0; r < m; ++r) mat[r + c * m] = 1.0;
    }
    for (int r = 0; r < m; ++r) lhs[r] = 0.0;

    double start = omp_get_wtime();
    dmvm_v1(n, m, lhs, rhs, mat);
    double end = omp_get_wtime();

    printf("Version 1 (Implicit Barrier) Time: %.6f s\n", end - start);
    printf("Sample Result: lhs[0] = %.1f\n", lhs[0]);

    free(mat); free(rhs); free(lhs);
    return 0;
}