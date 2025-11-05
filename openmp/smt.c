#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define PI 3.14159265358979323846

double an_func(double x, double y, double z, double t, double ax, double ay, double az, double a_t) {
    return sin(2 * PI * ax * x) * sin(PI * ay * y) * sin(PI * az * z) * cos(a_t * t + 2 * PI);
}

double check_layer(int n, double u[n][n][n], double ax, double ay, double az, double a_t, double hx, double hy, double hz, double t) {
    double max_dev = 0; //max deviation

    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            for (int k = 0; k <= n; k++) {
                double target = an_func(i * hx, j * hy, k * hz, t, ax, ay, az, a_t);
                double tmp_dev = fabs(u[i != n ? i : 0][j != n ? j : 0][k != n ? k : 0] - target);

                if (tmp_dev > max_dev) {
                    max_dev = tmp_dev;
                }
            }
        }
    }

    return max_dev;
}

double laplas(int n, double u[n][n][n], double sqhx, double sqhy, double sqhz, int i, int j, int k) {
    return (u[i - 1][j][k] - 2 * u[i][j][k] + u[i + 1][j][k]) / sqhx +
           (u[i][j - 1][k] - 2 * u[i][j][k] + u[i][j + 1][k]) / sqhy + 
           (u[i][j][k - 1] - 2 * u[i][j][k] + u[i][j][k + 1]) / sqhz;
}

double laplas_v2(int n, double u[n][n][n], double sqhx, double sqhy, double sqhz, int i, int j, int k) {
    int iplus = i < n - 1 ? i + 1 : 0;
    int jplus = j < n - 1 ? j + 1 : 0;
    int kplus = k < n - 1 ? k + 1 : 0;
    int iminus = i > 0 ? i - 1 : n - 1;
    int jminus = j > 0 ? j - 1 : n - 1;
    int kminus = k > 0 ? k - 1 : n - 1;

    return (u[iminus][j][k] - 2 * u[i][j][k] + u[iplus][j][k]) / sqhx +
           (u[i][jminus][k] - 2 * u[i][j][k] + u[i][jplus][k]) / sqhy + 
           (u[i][j][kminus] - 2 * u[i][j][k] + u[i][j][kplus]) / sqhz;
}

void next_iter(int n, double arr[n][n][n], double cur[n][n][n], double sqhx, double sqhy, double sqhz, double sqathau) {
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            for (int k = 1; k < n - 1; k++) {
                arr[i][j][k] = sqathau * laplas(n, cur, sqhx, sqhy, sqhz, i, j, k) +
                               2 * cur[i][j][k] - arr[i][j][k];
            }
        }
    }

    for (int j = 1; j < n - 1; j++) {
        for (int k = 1; k < n - 1; k++) {
            arr[0][j][k] = sqathau * laplas_v2(n, cur, sqhx, sqhy, sqhz, 0, j, k) +
                           2 * cur[0][j][k] - arr[0][j][k];
            arr[n - 1][j][k] = sqathau * laplas_v2(n, cur, sqhx, sqhy, sqhz, n - 1, j, k) +
                               2 * cur[n - 1][j][k] - arr[n - 1][j][k];
        }
    }

    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            arr[i][j][n - 1] = sqathau * laplas_v2(n, cur, sqhx, sqhy, sqhz, i, j, n - 1) +
                               2 * cur[i][j][n - 1] - arr[i][j][n - 1];
        }
    }

    for (int i = 1; i < n - 1; i++) {
        for (int k = 1; k < n - 1; k++) {
            arr[i][n - 1][k] = sqathau * laplas_v2(n, cur, sqhx, sqhy, sqhz, i, n - 1, k) +
                               2 * cur[i][n - 1][k] - arr[i][n - 1][k];
        }
    }

    for (int k = 1; k < n - 1; k++) {
        arr[0][n - 1][k] = sqathau * laplas_v2(n, cur, sqhx, sqhy, sqhz, 0, n - 1, k) +
                           2 * cur[0][n - 1][k] - arr[0][n - 1][k];
        arr[n - 1][n - 1][k] = sqathau * laplas_v2(n, cur, sqhx, sqhy, sqhz, n - 1, n - 1, k) +
                               2 * cur[n - 1][n - 1][k] - arr[n - 1][n - 1][k];
    }

    for (int j = 1; j < n - 1; j++) {
        arr[0][j][n - 1] = sqathau * laplas_v2(n, cur, sqhx, sqhy, sqhz, 0, j, n - 1) +
                           2 * cur[0][j][n - 1] - arr[0][j][n - 1];
        arr[n - 1][j][n - 1] = sqathau * laplas_v2(n, cur, sqhx, sqhy, sqhz, n - 1, j, n - 1) +
                               2 * cur[n - 1][j][n - 1] - arr[n - 1][j][n - 1];
    }

    for (int i = 1; i < n - 1; i++) {
        arr[i][n - 1][n - 1] = sqathau * laplas_v2(n, cur, sqhx, sqhy, sqhz, i, n - 1, n - 1) +
                               2 * cur[i][n - 1][n - 1] - arr[i][n - 1][n - 1];
    }

    arr[0][n - 1][n - 1] = sqathau * laplas_v2(n, cur, sqhx, sqhy, sqhz, 0, n - 1, n - 1) +
                           2 * cur[0][n - 1][n - 1] - arr[0][n - 1][n - 1];

    arr[n - 1][n - 1][n - 1] = sqathau * laplas_v2(n, cur, sqhx, sqhy, sqhz, n - 1, n - 1, n - 1) +
                               2 * cur[n - 1][n - 1][n - 1] - arr[n - 1][n - 1][n - 1];
}

int main (int argc, char** argv) {
    double lx, ly, lz;
    sscanf(argv[1], "%lf", &lx);
    sscanf(argv[2], "%lf", &ly);
    sscanf(argv[3], "%lf", &lz);

    double ax = 1 / lx;
    double ay = 1 / ly;
    double az = 1 / lz;

    double a_t = (sqrt(4 / (lx * lx) + 1 / (ly * ly) + 1 / (lz * lz))) / 2;

    int n = atoi(argv[4]);

    void* ptr_u0 = calloc(n * n * n, sizeof(double));
    void* ptr_u1 = calloc(n * n * n, sizeof(double));

    double (*u0)[n][n] = ptr_u0;
    double (*u1)[n][n] = ptr_u1;

    double hx = lx / n;
    double hy = ly / n;
    double hz = lz / n;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                u0[i][j][k] = an_func(i * hx, j * hy, k * hz, 0, ax, ay, az, a_t);
            }
        }
    }

    free(ptr_u0);
    free(ptr_u1);
}