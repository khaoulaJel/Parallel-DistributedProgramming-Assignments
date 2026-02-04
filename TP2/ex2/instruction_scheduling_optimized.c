#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000LL

int main() {
    double a = 1.1, b = 1.2;
    double x = 0.0, y = 0.0;
    clock_t start, end;
    
    double ab = a * b;  // compute once outside loop

    start = clock();
    for (long long i = 0; i < N; i++) {
        x += ab;
        y += ab;
    }
    end = clock();
    
    printf("x = %f, y = %f, time = %.6f s\n", 
           x, y, (double)(end - start) / CLOCKS_PER_SEC);
    return 0;
}
