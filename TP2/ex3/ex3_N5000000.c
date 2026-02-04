#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 5000000LL

void add_noise(double *a) {
    a[0] = 1.0;
    for (long long i = 1; i < N; i++) {
        a[i] = a[i-1] * 1.0000001;
    }
}

void init_b(double *b) {
    for (long long i = 0; i < N; i++) {
        b[i] = i * 0.5;
    }
}

void compute_addition(double *a, double *b, double *c) {
    for (long long i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

double reduction(double *c) {
    double sum = 0.0;
    for (long long i = 0; i < N; i++) {
        sum += c[i];
    }
    return sum;
}

int main() {
    double *a = malloc(N * sizeof(double));
    double *b = malloc(N * sizeof(double));
    double *c = malloc(N * sizeof(double));
    
    if (!a || !b || !c) {
        fprintf(stderr, "Malloc failed\n");
        return 1;
    }
    
    clock_t start, end;
    double t_add_noise = 0.0, t_init = 0.0, t_compute = 0.0, t_reduction = 0.0;
    
    // add_noise – strictly sequential
    start = clock();
    add_noise(a);
    end = clock();
    t_add_noise = (double)(end - start) / CLOCKS_PER_SEC;
    
    // init_b – parallelizable
    start = clock();
    init_b(b);
    end = clock();
    t_init = (double)(end - start) / CLOCKS_PER_SEC;
    
    // compute_addition – parallelizable
    start = clock();
    compute_addition(a, b, c);
    end = clock();
    t_compute = (double)(end - start) / CLOCKS_PER_SEC;
    
    // reduction – parallelizable
    start = clock();
    double sum = reduction(c);
    end = clock();
    t_reduction = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Sum = %.15f\n", sum);
    printf("Time add_noise:        %.6f s\n", t_add_noise);
    printf("Time init_b:           %.6f s\n", t_init);
    printf("Time compute_addition: %.6f s\n", t_compute);
    printf("Time reduction:        %.6f s\n", t_reduction);
    
    free(a); free(b); free(c);
    return 0;
}
