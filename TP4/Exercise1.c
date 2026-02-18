#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif 

#define N 1000000

int main(){

    double *A = malloc(N * sizeof(double));
    if (A == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    double sum = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    double max;

    // Initialization
    srand(0);
    for (int i = 0; i < N; i++)
        A[i] = (double)rand() / RAND_MAX;
    
    sum = 0.0;
    max = A[0];
    
    #pragma omp parallel sections
    {
        #pragma omp section 
        {
            double local_sum = 0.0;  
            for (int i = 0; i < N; i++) {
                local_sum += A[i];
            }
            #pragma omp atomic 
            sum += local_sum;  
        }
        
        #pragma omp section 
        {
            double local_max = A[0];
            for (int i = 0; i < N; i++) {
                if (A[i] > local_max)  
                    local_max = A[i];
            }
            #pragma omp critical
            {
                if (local_max > max)
                    max = local_max;
            } 
        }
    }
    
    mean = sum / N;

    #pragma omp parallel 
    {
        double local_stddev = 0.0;  
        #pragma omp for  
        for (int i = 0; i < N; i++) {
            local_stddev += (A[i] - mean) * (A[i] - mean);
        }
        #pragma omp atomic
        stddev += local_stddev;
    }

    stddev = sqrt(stddev / N);
    
    printf("Sum     = %f\n", sum);
    printf("Max     = %f\n", max);
    printf("Std Dev = %f\n", stddev);

    free(A);
    return 0;
}