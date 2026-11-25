#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

typedef struct {
    uint64_t state;
} XorShift64Star;

static void rng_init(XorShift64Star *rng, uint64_t seed) {
    rng->state = seed ? seed : 0x9e3779b97f4a7c15ULL;
}

static uint64_t rng_next(XorShift64Star *rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 2685821657736338717ULL;
}

static double rng_uniform01(XorShift64Star *rng) {
    return (double)(rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

static double rng_uniformPM1(XorShift64Star *rng) {
    return 2.0 * rng_uniform01(rng) - 1.0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s N [threads]\n", argv[0]);
        return 1;
    }
    long long N = atoll(argv[1]);
    int threads = (argc >= 3) ? atoi(argv[2]) : omp_get_max_threads();
    if (N <= 0) {
        fprintf(stderr, "N must be positive\n");
        return 1;
    }
    if (threads <= 0) {
        threads = omp_get_max_threads();
    }
    double inside = 0.0;
    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(threads) reduction(+:inside)
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        uint64_t seed = 0x123456789abcdef0ULL ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(tid + 1));
        XorShift64Star rng;
        rng_init(&rng, seed);
        long long chunk = N / nt;
        long long begin = (long long)tid * chunk;
        long long end = (tid == nt - 1) ? N : begin + chunk;
        long long local_inside = 0;
        for (long long i = begin; i < end; ++i) {
            double x = rng_uniformPM1(&rng);
            double y = rng_uniformPM1(&rng);
            if (x * x + y * y <= 1.0) {
                ++local_inside;
            }
        }
        inside += (double)local_inside;
    }
    double t1 = omp_get_wtime();
    long long inside_ll = (long long)inside;
    double pi = 4.0 * inside / (double)N;
    double error = fabs(pi - 3.14159265358979323846);
    printf("N=%lld threads=%d inside=%lld pi=%.10f error=%.6e time=%.6f s\n", N, threads, inside_ll, pi, error, t1 - t0);
    return 0;
}
