#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000

void add_noise(double *a) {
    a[0] = 1.0;
    for (int i = 1; i < N; i++) {
        a[i] = a[i-1] * 1.0000001;
    }
}

void init_b(double *b) {
    for (int i = 0; i < N; i++) {
        b[i] = i * 0.5;
    }
}

void compute_addition(double *a, double *b, double *c) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

double reduction(double *c) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += c[i];
    }
    return sum;
}

int main() {
    double *a = malloc(N * sizeof(double));
    double *b = malloc(N * sizeof(double));
    double *c = malloc(N * sizeof(double));
    
    clock_t start, end;
    double time_add_noise, time_init_b, time_compute, time_reduction;
    
    start = clock();
    add_noise(a);
    end = clock();
    time_add_noise = (double)(end - start) / CLOCKS_PER_SEC;
    
    start = clock();
    init_b(b);
    end = clock();
    time_init_b = (double)(end - start) / CLOCKS_PER_SEC;
    
    start = clock();
    compute_addition(a, b, c);
    end = clock();
    time_compute = (double)(end - start) / CLOCKS_PER_SEC;
    
    start = clock();
    double sum = reduction(c);
    end = clock();
    time_reduction = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Sum = %f\n", sum);
    printf("Time add_noise: %.6f s\n", time_add_noise);
    printf("Time init_b: %.6f s\n", time_init_b);
    printf("Time compute_addition: %.6f s\n", time_compute);
    printf("Time reduction: %.6f s\n", time_reduction);
    
    free(a);
    free(b);
    free(c);
    return 0;
}
