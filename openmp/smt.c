#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define PI 3.14159265358979323846

double an_func(double x, double y, double z, double t, double ax, double ay, double az, double a_t) {
    return sin(2 * PI * ax * x) * sin(PI * ay * y) * sin(PI * az * z) * cos(a_t * t + 2 * PI);
}

double check_layer(int n, double u[n + 1][n + 1][n + 1], double ax, double ay, double az, double a_t, double hx, double hy, double hz, double t, int threads) {
    double max_dev = 0; //max deviation

    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
#pragma omp parallel for  num_threads(threads) reduction(max:max_dev)
            for (int k = 0; k <= n; k++) {
                double target = an_func(i * hx, j * hy, k * hz, t, ax, ay, az, a_t);
                double tmp_dev = fabs(u[i][j][k] - target);

                if (tmp_dev > max_dev) {
                    max_dev = tmp_dev;
                }
            }
        }
    }

    return max_dev;
}

double laplas(int n, double u[n + 1][n + 1][n +1], double sqhx, double sqhy, double sqhz, int i, int j, int k) {
    return (u[i - 1][j][k] - 2 * u[i][j][k] + u[i + 1][j][k]) / sqhx +
           (u[i][j - 1][k] - 2 * u[i][j][k] + u[i][j + 1][k]) / sqhy + 
           (u[i][j][k - 1] - 2 * u[i][j][k] + u[i][j][k + 1]) / sqhz;
}

double laplas_v2(int n, double u[n + 1][n + 1][n + 1], double sqhx, double sqhy, double sqhz, int i, int j, int k) {
    int iplus = i < n ? i + 1 : 1;
    int iminus = i > 0 ? i - 1 : n - 1;

    return (u[iminus][j][k] - 2 * u[i][j][k] + u[iplus][j][k]) / sqhx +
           (u[i][j - 1][k] - 2 * u[i][j][k] + u[i][j + 1][k]) / sqhy + 
           (u[i][j][k - 1] - 2 * u[i][j][k] + u[i][j][k + 1]) / sqhz;
}

void next_iter(int n, double arr[n + 1][n + 1][n + 1], double cur[n + 1][n + 1][n + 1], double sqhx, double sqhy, double sqhz, double sqathau, int t) {
#pragma omp parallel num_threads(t)
{
    #pragma omp for collapse(3) nowait
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < n; j++) {
            for (int k = 1; k < n; k++) {
                arr[i][j][k] = sqathau * laplas(n, cur, sqhx, sqhy, sqhz, i, j, k) +
                               2 * cur[i][j][k] - arr[i][j][k];
            }
        }
    }

    #pragma omp for collapse(2) nowait
    for (int j = 1; j < n; j++) {
        for (int k = 1; k < n; k++) {
            arr[0][j][k] = sqathau * laplas_v2(n, cur, sqhx, sqhy, sqhz, 0, j, k) +
                           2 * cur[0][j][k] - arr[0][j][k];
            arr[n][j][k] = sqathau * laplas_v2(n, cur, sqhx, sqhy, sqhz, n, j, k) +
                               2 * cur[n][j][k] - arr[n][j][k];
        }
    }
}
}

int main (int argc, char** argv) {
    if (argc < 8) {
        printf("Not enough arguments.\n");
        exit(1);
    }
    double lx, ly, lz;
    sscanf(argv[1], "%lf", &lx);
    sscanf(argv[2], "%lf", &ly);
    sscanf(argv[3], "%lf", &lz);

    double ax = 1 / lx;
    double ay = 1 / ly;
    double az = 1 / lz;

    double a_t = (sqrt(4 / (lx * lx) + 1 / (ly * ly) + 1 / (lz * lz))) / 2;

    int n = atoi(argv[4]);

    double time;
    sscanf(argv[5], "%lf", &time);
    int steps = atoi(argv[6]);
    int threads = atoi(argv[7]);

    double sqathau = (time / steps) * (time / steps) / (4 * PI * PI);

    void* ptr_u0 = malloc(sizeof(double) * ((n + 1) * (n + 1) * (n + 1)));
    void* ptr_u1 = malloc(sizeof(double) * ((n + 1) * (n + 1) * (n + 1)));

    double (*u0)[n + 1][n + 1] = ptr_u0;
    double (*u1)[n + 1][n + 1] = ptr_u1;

    double hx = lx / n;
    double hy = ly / n;
    double hz = lz / n;

#pragma omp parallel for collapse(3) num_threads(threads)
    for (int i = 0; i < n + 1; i++) {
        for (int j = 0; j < n + 1; j++) {
            for (int k = 0; k < n + 1; k++) {
                u0[i][j][k] = an_func(i * hx, j * hy, k * hz, 0, ax, ay, az, a_t);
                u1[i][j][k] = u0[i][j][k];
            }
        }
    }
    
    next_iter(n, u1, u0, hx * hx, hy * hy, hz * hz, sqathau / 2, threads);

    double (*arr)[n + 1][n + 1] = u0;
    double (*cur)[n + 1][n + 1] = u1;
    double (*tmp)[n + 1][n + 1];

    double start_time, end_time;
    start_time = omp_get_wtime();
    for (int i = 2; i <= steps; i++) {
        next_iter(n, arr, cur, hx * hx, hy * hy, hz * hz, sqathau, threads);
        tmp = cur;
        cur = arr;
        arr = tmp;
    }
    end_time = omp_get_wtime();

    printf("Max deviation u%d: %.12lf\n", steps, check_layer(n, cur, ax, ay, az, a_t, hx, hy, hz, time, threads));
    printf("Time spent: %0.6lf\n", end_time - start_time);

    free(ptr_u0);
    free(ptr_u1);
}