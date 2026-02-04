#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 256

void generate_noise(double *noise) {
    noise[0] = 1.0;
    for (int i = 1; i < N; i++) {
        noise[i] = noise[i-1] * 1.0000001;
    }
}

void init_matrix(double *M) {
    for (int i = 0; i < N*N; i++) {
        M[i] = (double)(i % 100) * 0.01;
    }
}

void matmul(double *A, double *B, double *C, double *noise) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = noise[i];
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

int main() {
    double *A = malloc(N*N * sizeof(double));
    double *B = malloc(N*N * sizeof(double));
    double *C = malloc(N*N * sizeof(double));
    double *noise = malloc(N * sizeof(double));
    
    clock_t start, end;
    double time_noise, time_init, time_matmul;
    
    start = clock();
    generate_noise(noise);
    end = clock();
    time_noise = (double)(end - start) / CLOCKS_PER_SEC;
    
    start = clock();
    init_matrix(A);
    init_matrix(B);
    end = clock();
    time_init = (double)(end - start) / CLOCKS_PER_SEC;
    
    start = clock();
    matmul(A, B, C, noise);
    end = clock();
    time_matmul = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("C[0] = %f\n", C[0]);
    printf("Time generate_noise: %.6f s\n", time_noise);
    printf("Time init_matrix: %.6f s\n", time_init);
    printf("Time matmul: %.6f s\n", time_matmul);
    
    free(A);
    free(B);
    free(C);
    free(noise);
    return 0;
}
