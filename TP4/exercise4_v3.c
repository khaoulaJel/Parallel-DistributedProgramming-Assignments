#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void dmvm_v3(int n, int m, double *lhs, double *rhs, double *mat) {
    #pragma omp parallel
    {
        // static: iterations are divided equally among threads
        #pragma omp for schedule(static) nowait reduction(+:lhs[:m])
        for (int c = 0; c < n; ++c) {
            int offset = m * c;
            for (int r = 0; r < m; ++r) {
                lhs[r] += mat[r + offset] * rhs[c];
            }
        }
    }
}

int main() {
    const int n = 40000; const int m = 600;
    double *mat = malloc(n * m * sizeof(double));
    double *rhs = malloc(n * sizeof(double));
    double *lhs = malloc(m * sizeof(double));

    for (int c = 0; c < n; ++c) { rhs[c] = 1.0; for (int r = 0; r < m; ++r) mat[r + c * m] = 1.0; }
    for (int r = 0; r < m; ++r) lhs[r] = 0.0;

    double start = omp_get_wtime();
    dmvm_v3(n, m, lhs, rhs, mat);
    double end = omp_get_wtime();

    printf("Version 3 (Static + Nowait) Time: %.6f s\n", end - start);
    free(mat); free(rhs); free(lhs);
    return 0;
}