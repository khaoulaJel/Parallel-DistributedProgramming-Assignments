#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000ULL          // 100 million elements

int main(void) {
    int *a = malloc(N * sizeof(int));
    if (!a) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    long long sum = 0;
    volatile long long sink;           // prevent optimization away

    // Warm-up + initialize array
    for (size_t i = 0; i < N; i++) {
        a[i] = 1;
    }

    // Warm-up loops (bring data into cache, stabilize frequency)
    for (int rep = 0; rep < 3; rep++) {
        sum = 0;
        int limit = N - (N % 12);
        for (size_t i = 0; i < limit; i += 12) {
            sum += a[i + 0] + a[i + 1] + a[i + 2] + a[i + 3] + a[i + 4] + a[i + 5] + a[i + 6] + a[i + 7] + a[i + 8] + a[i + 9] + a[i + 10] + a[i + 11];
        }
        for (size_t i = limit; i < N; i++) {
            sum += a[i];
        }
    }

    // Real measurement ── take the best of 5 runs
    double best_time_ms = 1e9;

    for (int rep = 0; rep < 5; rep++) {
        sum = 0;

        struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);

        int limit = N - (N % 12);
        for (size_t i = 0; i < limit; i += 12) {
            sum += a[i + 0] + a[i + 1] + a[i + 2] + a[i + 3] + a[i + 4] + a[i + 5] + a[i + 6] + a[i + 7] + a[i + 8] + a[i + 9] + a[i + 10] + a[i + 11];
        }
        for (size_t i = limit; i < N; i++) {
            sum += a[i];
        }

        clock_gettime(CLOCK_MONOTONIC, &t2);

        double dt_ms = (t2.tv_sec - t1.tv_sec) * 1000.0 +
                       (t2.tv_nsec - t1.tv_nsec) / 1000000.0;

        if (dt_ms < best_time_ms) best_time_ms = dt_ms;
    }

    sink = sum;     // force use of result

    printf("U=%2d | Type=%-6s | Sum=%lld | Time=%.3f ms | %s\n",
           12, "int", sink, best_time_ms, "4 bytes");

    free(a);
    return 0;
}
