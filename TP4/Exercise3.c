#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 10000000

double task_light(int n) {
    double x = 0.0;
    for (int i = 0; i < n; i++) {
        x += sin(i * 0.001);
    }
    return x;
}

double task_moderate(int n) {
    double x = 0.0;
    for (int i = 0; i < 5*n; i++) {
        x += sqrt(i * 0.5) * cos(i * 0.001);
    }
    return x;
}

double task_heavy(int n) {
    double x = 0.0;
    for (int i = 0; i < 20*n; i++) {
        x += sqrt(i * 0.5) * cos(i * 0.001) * sin(i * 0.0001);
    }
    return x;
}

int main() {
    double start, end;
    double total = 0.0;

    start = omp_get_wtime();

    total += task_light(N);
    total += task_moderate(N);
    total += task_heavy(N);

    end = omp_get_wtime();
    
    printf("Total Result: %f\n", total); 
    printf("Sequential execution time = %.6f seconds\n", end - start);
    return 0;
}