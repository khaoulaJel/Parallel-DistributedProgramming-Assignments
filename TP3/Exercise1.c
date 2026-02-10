#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        printf("Hello from thread %d\n", thread_id);
    }
    int num_threads = omp_get_max_threads();
    
        
    printf("Parallel execution of hello_world with %d\n", num_threads);
    return 0;
}