#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int nodes;
    double *temperature_rise;  /* u - c, including a zero boundary */
    double *scaled_source;     /* h^2 f(x, y) */
} Plate;

/* This source gives u(x,y) = c + x(1-x)y(1-y). */
static double source(double x, double y)
{
    return -2.0 * (x * (1.0 - x) + y * (1.0 - y));
}

static void fill_source(Plate plate)
{
    double h = 1.0 / (plate.nodes - 1);
    for (int row = 1; row < plate.nodes - 1; ++row) {
        for (int col = 1; col < plate.nodes - 1; ++col) {
            size_t index = (size_t)row * plate.nodes + col;
            plate.scaled_source[index] = h * h * source(col * h, row * h);
        }
    }
}

static int min_int(int a, int b)
{
    return a < b ? a : b;
}

static double update_block(Plate plate, int block_row, int block_col, int block_size)
{
    int first_row = 1 + block_row * block_size;
    int first_col = 1 + block_col * block_size;
    int end_row = first_row + min_int(block_size, plate.nodes - 1 - first_row);
    int end_col = first_col + min_int(block_size, plate.nodes - 1 - first_col);
    double *u = plate.temperature_rise;
    double max_change = 0.0;

    for (int row = first_row; row < end_row; ++row) {
        for (int col = first_col; col < end_col; ++col) {
            size_t i = (size_t)row * plate.nodes + col;
            double next = 0.25 * (u[i - plate.nodes] + u[i + plate.nodes]
                                + u[i - 1] + u[i + 1] - plate.scaled_source[i]);
            max_change = fmax(max_change, fabs(next - u[i]));
            u[i] = next;
        }
    }
    return max_change;
}

static double gauss_seidel_sweep(Plate plate, int block_size)
{
    int blocks = 1 + (plate.nodes - 3) / block_size;
    int threads = min_int(blocks, omp_get_max_threads());
    double max_change = 0.0;

    #pragma omp parallel num_threads(threads) reduction(max:max_change)
    {
        for (int wave = 0; wave <= 2 * (blocks - 1); ++wave) {
            int last_row = min_int(wave, blocks - 1);
            int first_row = wave - last_row;
            /* The implicit barrier completes the top and left dependencies
               before any thread starts the next wave. */
            #pragma omp for schedule(static)
            for (int row = first_row; row <= last_row; ++row) {
                double change = update_block(plate, row, wave - row, block_size);
                max_change = fmax(max_change, change);
            }
        }
    }
    return max_change;
}

static int positive_int(const char *text)
{
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end || value < 1 || value > INT_MAX) {
        return 0;
    }
    return (int)value;
}

static double finite_number(const char *text)
{
    char *end;
    errno = 0;
    double value = strtod(text, &end);
    return errno || end == text || *end || !isfinite(value) ? NAN : value;
}

static int print_temperatures(Plate plate, double boundary)
{
    for (int row = 0; row < plate.nodes; ++row) {
        for (int col = 0; col < plate.nodes; ++col) {
            size_t i = (size_t)row * plate.nodes + col;
            char separator = col == plate.nodes - 1 ? '\n' : ' ';
            if (printf("%.17g%c", boundary + plate.temperature_rise[i], separator) < 0) {
                return 0;
            }
        }
    }
    return fflush(stdout) == 0;
}

int main(int argc, char **argv)
{
    if (argc < 4 || argc > 6) {
        fprintf(stderr, "Usage: %s nodes_per_side block_size tolerance "
                        "[boundary_temperature=0] [max_sweeps=100000]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int nodes = positive_int(argv[1]);
    int block_size = positive_int(argv[2]);
    double tolerance = finite_number(argv[3]);
    double boundary = argc >= 5 ? finite_number(argv[4]) : 0.0;
    int max_sweeps = argc == 6 ? positive_int(argv[5]) : 100000;
    if (nodes < 3 || !block_size || !(tolerance > 0) || !isfinite(boundary) || !max_sweeps) {
        fprintf(stderr, "Need at least 3 nodes per side, positive block size, tolerance "
                        "and sweep limit, and a finite boundary temperature.\n");
        return EXIT_FAILURE;
    }
    if ((size_t)nodes > SIZE_MAX / sizeof(double) / (size_t)nodes) {
        fprintf(stderr, "Grid dimensions exceed addressable memory.\n");
        return EXIT_FAILURE;
    }
    size_t cells = (size_t)nodes * nodes;
    Plate plate = {nodes, calloc(cells, sizeof(double)), malloc(cells * sizeof(double))};
    int status = EXIT_FAILURE;
    if (!plate.temperature_rise || !plate.scaled_source) {
        fprintf(stderr, "Cannot allocate the grid.\n");
    } else {
        fill_source(plate);
        double start = omp_get_wtime();
        int sweeps = 0;
        double max_change;
        do {
            max_change = gauss_seidel_sweep(plate, block_size);
            ++sweeps;
        } while (max_change > tolerance && sweeps < max_sweeps);
        fprintf(stderr, "sweeps=%d max_change=%.9g solve_seconds=%.6f\n",
                sweeps, max_change, omp_get_wtime() - start);
        if (max_change > tolerance) {
            fprintf(stderr, "Sweep limit reached before convergence.\n");
        } else if (print_temperatures(plate, boundary)) {
            status = EXIT_SUCCESS;
        } else {
            fprintf(stderr, "Cannot write the temperature field.\n");
        }
    }
    free(plate.scaled_source);
    free(plate.temperature_rise);
    return status;
}
