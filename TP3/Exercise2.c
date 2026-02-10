#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif


static long num_steps = 100000;
double step;

int main () {
    int i; double x, pi, sum = 0.0;
    step = 1.0 / (double) num_steps;
    double time = omp_get_wtime();

    #pragma omp parallel private(x,i)
    {
        int id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        double local_sum = 0.0;
        for (i = id; i < num_steps; i += num_threads) {
            x = (i + 0.5) * step;
            local_sum += 4.0 / (1.0 + x * x);
        }
        #pragma omp critical
            sum += local_sum;
    }
    pi = step * sum;
    time = omp_get_wtime() - time;
    printf("pi is approximately: %lf\n", pi);
    printf("Time taken: %lf seconds\n", time);
    return 0;
}