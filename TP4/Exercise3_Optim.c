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

double task_heavy(int start_idx, int end_idx) {
    double x = 0.0;
    for (int i = start_idx; i < end_idx; i++) {
        x += sqrt(i * 0.5) * cos(i * 0.001) * sin(i * 0.0001);
    }
    return x;
}

int main() {
    double start, end;
    double total_res = 0.0;

    start = omp_get_wtime();

    #pragma omp parallel sections reduction(+:total_res)
    {
        #pragma omp section
        total_res += task_light(N);

        #pragma omp section
        total_res += task_moderate(N);


        #pragma omp section
        total_res += task_heavy(0, 5*N);

        #pragma omp section
        total_res += task_heavy(5*N, 10*N);

        #pragma omp section
        total_res += task_heavy(10*N, 15*N);

        #pragma omp section
        total_res += task_heavy(15*N, 20*N);
    }

    end = omp_get_wtime();
    printf("Result: %f\n", total_res); 
    printf("Optimized Parallel Time = %.6f seconds\n", end - start);

    return 0;
}