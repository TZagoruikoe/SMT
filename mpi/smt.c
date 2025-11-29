#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define PI 3.14159265358979323846

typedef struct {
    int count_elem;
    int start_elem;
} for_coord_t;

int min(int a, int b);
inline int min(int a, int b) {
    if (a <= b) {
        return a;
    }
    else {
        return b;
    }
}

double an_func(double x, double y, double z, double t, double ax, double ay, double az, double a_t) {
    return sin(2 * PI * ax * x) * sin(PI * ay * y) * sin(PI * az * z) * cos(a_t * t + 2 * PI);
}

double check_layer(for_coord_t offset[3], double u[offset[0].count_elem + 2][offset[1].count_elem + 2][offset[2].count_elem + 2], double ax, double ay, double az, double a_t, double hx, double hy, double hz, double t) {
    double max_dev = 0; //max deviation

    for (int i = 1; i <= offset[0].count_elem; i++) {
        for (int j = 1; j <= offset[1].count_elem; j++) {
            for (int k = 1; k <= offset[2].count_elem; k++) {
                double target = an_func((i + offset[0].start_elem - 1) * hx, (j + offset[1].start_elem - 1) * hy, (k + offset[2].start_elem - 1) * hz, t, ax, ay, az, a_t);
                double tmp_dev = fabs(u[i][j][k] - target);

                if (tmp_dev > max_dev) {
                    max_dev = tmp_dev;
                }
            }
        }
    }

    return max_dev;
}

double laplas(for_coord_t offset[3], double u[offset[0].count_elem + 2][offset[1].count_elem + 2][offset[2].count_elem + 2], 
              double sqhx, double sqhy, double sqhz, int i, int j, int k) {
    return (u[i - 1][j][k] - 2 * u[i][j][k] + u[i + 1][j][k]) / sqhx +
           (u[i][j - 1][k] - 2 * u[i][j][k] + u[i][j + 1][k]) / sqhy + 
           (u[i][j][k - 1] - 2 * u[i][j][k] + u[i][j][k + 1]) / sqhz;
}

void next_iter(int n, MPI_Comm comm_cart, MPI_Datatype xy_side, MPI_Datatype xz_side, MPI_Datatype yz_side, 
               for_coord_t offset[3], double arr[offset[0].count_elem + 2][offset[1].count_elem + 2][offset[2].count_elem + 2], 
               double cur[offset[0].count_elem + 2][offset[1].count_elem + 2][offset[2].count_elem + 2], 
               double sqhx, double sqhy, double sqhz, double sqathau) {
    int dim_x_right, dim_x_left;
    int dim_y_right, dim_y_left;
    int dim_z_right, dim_z_left;

    MPI_Cart_shift(comm_cart, 0, 1, &dim_x_left, &dim_x_right);
    MPI_Cart_shift(comm_cart, 1, 1, &dim_y_left, &dim_y_right);
    MPI_Cart_shift(comm_cart, 2, 1, &dim_z_left, &dim_z_right);

    MPI_Request reqs[12];
    //x - fix
    if (offset[0].start_elem == 0) {
        MPI_Isend(&cur[2][1][1], 1, yz_side, dim_x_left, 0, comm_cart, &reqs[0]);
    }
    else {
        MPI_Isend(&cur[1][1][1], 1, yz_side, dim_x_left, 0, comm_cart, &reqs[0]);
    }
    MPI_Irecv(&cur[offset[0].count_elem + 1][1][1], 1, yz_side, dim_x_right, 0, comm_cart, &reqs[1]);
    MPI_Irecv(&cur[0][1][1], 1, yz_side, dim_x_left, 1, comm_cart, &reqs[2]);

    if (offset[0].start_elem + offset[0].count_elem == n + 1) {
        MPI_Isend(&cur[offset[0].count_elem - 1][1][1], 1, yz_side, dim_x_right, 1, comm_cart, &reqs[3]);
    }
    else {
        MPI_Isend(&cur[offset[0].count_elem][1][1], 1, yz_side, dim_x_right, 1, comm_cart, &reqs[3]);
    }

    //y - fix
    MPI_Isend(&cur[1][1][1], 1, xz_side, dim_y_left, 2, comm_cart, &reqs[4]);
    MPI_Irecv(&cur[1][offset[1].count_elem + 1][1], 1, xz_side, dim_y_right, 2, comm_cart, &reqs[5]);
    MPI_Irecv(&cur[1][0][1], 1, xz_side, dim_y_left, 3, comm_cart, &reqs[6]);
    MPI_Isend(&cur[1][offset[1].count_elem][1], 1, xz_side, dim_y_right, 3, comm_cart, &reqs[7]);

    //z - fix
    MPI_Isend(&cur[1][1][1], 1, xy_side, dim_z_left, 4, comm_cart, &reqs[8]);
    MPI_Irecv(&cur[1][1][offset[2].count_elem + 1], 1, xy_side, dim_z_right, 4, comm_cart, &reqs[9]);
    MPI_Irecv(&cur[1][1][0], 1, xy_side, dim_z_left, 5, comm_cart, &reqs[10]);
    MPI_Isend(&cur[1][1][offset[2].count_elem], 1, xy_side, dim_z_right, 5, comm_cart, &reqs[11]);
    
    for (int i = 2; i < offset[0].count_elem; i++) {
        for (int j = 2; j < offset[1].count_elem; j++) {
            for (int k = 2; k < offset[2].count_elem; k++) {
                arr[i][j][k] = sqathau * laplas(offset, cur, sqhx, sqhy, sqhz, i, j, k) +
                               2 * cur[i][j][k] - arr[i][j][k];
            }
        }
    }

    MPI_Waitall(12, reqs, MPI_STATUS_IGNORE);

    for (int j = 2; j < offset[1].count_elem; j++) {
        for (int k = 2; k < offset[2].count_elem; k++) {
            arr[1][j][k] = sqathau * laplas(offset, cur, sqhx, sqhy, sqhz, 1, j, k) +
                           2 * cur[1][j][k] - arr[1][j][k];
            arr[offset[0].count_elem][j][k] = sqathau * laplas(offset, cur, sqhx, sqhy, sqhz, offset[0].count_elem, j, k) +
                            2 * cur[offset[0].count_elem][j][k] - arr[offset[0].count_elem][j][k];
        }
    }

    for (int i = 1; i <= offset[0].count_elem; i++) {
        for (int k = 2; k < offset[2].count_elem; k++) {
            if (offset[1].start_elem != 0) {
                arr[i][1][k] = sqathau * laplas(offset, cur, sqhx, sqhy, sqhz, i, 1, k) +
                           2 * cur[i][1][k] - arr[i][1][k];
            }
            if (offset[1].start_elem + offset[1].count_elem != n + 1) {
                arr[i][offset[1].count_elem][k] = sqathau * laplas(offset, cur, sqhx, sqhy, sqhz, i, offset[1].count_elem, k) +
                            2 * cur[i][offset[1].count_elem][k] - arr[i][offset[1].count_elem][k];
            }
        }
    }

    for (int i = 1; i <= offset[0].count_elem; i++) {
        for (int j = 2; j < offset[1].count_elem; j++) {
            if (offset[2].start_elem != 0) {
                arr[i][j][1] = sqathau * laplas(offset, cur, sqhx, sqhy, sqhz, i, j, 1) +
                               2 * cur[i][j][1] - arr[i][j][1];
            }
            if (offset[2].start_elem + offset[2].count_elem != n + 1) {
                arr[i][j][offset[2].count_elem] = sqathau * laplas(offset, cur, sqhx, sqhy, sqhz, i, j, offset[2].count_elem) +
                            2 * cur[i][j][offset[2].count_elem] - arr[i][j][offset[2].count_elem];
            }
        }
    }

    for (int i = 1; i <= offset[0].count_elem; i++) {
        if (offset[1].start_elem != 0) {
            if (offset[2].start_elem != 0) {
                arr[i][1][1] = sqathau * laplas(offset, cur, sqhx, sqhy, sqhz, i, 1, 1) +
                               2 * cur[i][1][1] - arr[i][1][1];
            }
            if (offset[2].start_elem + offset[2].count_elem != n + 1) {
                arr[i][1][offset[2].count_elem] = sqathau * laplas(offset, cur, sqhx, sqhy, sqhz, i, 1, offset[2].count_elem) +
                            2 * cur[i][1][offset[2].count_elem] - arr[i][1][offset[2].count_elem];
            }
        }
        if (offset[1].start_elem + offset[1].count_elem != n + 1) {
            if (offset[2].start_elem != 0) {
                arr[i][offset[1].count_elem][1] = sqathau * laplas(offset, cur, sqhx, sqhy, sqhz, i, offset[1].count_elem, 1) +
                               2 * cur[i][offset[1].count_elem][1] - arr[i][offset[1].count_elem][1];
            }
            if (offset[2].start_elem + offset[2].count_elem != n + 1) {
                arr[i][offset[1].count_elem][offset[2].count_elem] = sqathau * laplas(offset, cur, sqhx, sqhy, sqhz, i, offset[1].count_elem, offset[2].count_elem) +
                            2 * cur[i][offset[1].count_elem][offset[2].count_elem] - arr[i][offset[1].count_elem][offset[2].count_elem];
            }
        }
    }
}

for_coord_t offset_coord(int n, int coords, int dims) {
    for_coord_t offset;

    offset.start_elem = coords * n / dims + min(n % dims, coords);
    offset.count_elem = coords < n % dims ? n / dims + 1 : n / dims;

    return offset;
}

int main (int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 7) {
        if (rank == 0) {
            printf("Not enough arguments.\n");    
        }
        MPI_Finalize();
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

    double sqathau = (time / steps) * (time / steps) / (4 * PI * PI);

    double hx = lx / n;
    double hy = ly / n;
    double hz = lz / n;

    int dims[3] = {0, 0, 0};
    MPI_Dims_create(size, 3, dims);

    MPI_Comm comm_cart;
    int periods[3] = {1, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &comm_cart);

    int rank_cart;
    MPI_Comm_rank(comm_cart, &rank_cart);

    int coords[3] = {0, 0, 0}; //x, y, z
    MPI_Cart_coords(comm_cart, rank_cart, 3, coords);

    for_coord_t offset[3];
    offset[0] = offset_coord(n + 1, coords[0], dims[0]);
    offset[1] = offset_coord(n + 1, coords[1], dims[1]);
    offset[2] = offset_coord(n + 1, coords[2], dims[2]);

    MPI_Datatype y_row, xy_side, xz_side, yz_side;
    MPI_Type_vector(offset[1].count_elem, 1, offset[2].count_elem + 2, MPI_DOUBLE, &y_row);
    MPI_Type_commit(&y_row);

    int stride_x = (offset[2].count_elem + 2) * (offset[1].count_elem + 2);
    MPI_Type_create_hvector(offset[0].count_elem, 1, stride_x * sizeof(double), y_row, &xy_side);
    MPI_Type_commit(&xy_side);

    MPI_Type_vector(offset[0].count_elem, offset[2].count_elem, stride_x, MPI_DOUBLE, &xz_side);
    MPI_Type_commit(&xz_side);

    MPI_Type_vector(offset[1].count_elem, offset[2].count_elem, offset[2].count_elem + 2, MPI_DOUBLE, &yz_side);
    MPI_Type_commit(&yz_side);

    void* ptr_u0 = malloc(sizeof(double) * ((offset[0].count_elem + 2) * (offset[1].count_elem + 2) * (offset[2].count_elem + 2)));
    void* ptr_u1 = malloc(sizeof(double) * ((offset[0].count_elem + 2) * (offset[1].count_elem + 2) * (offset[2].count_elem + 2)));

    double (*u0)[offset[1].count_elem + 2][offset[2].count_elem + 2] = ptr_u0;
    double (*u1)[offset[1].count_elem + 2][offset[2].count_elem + 2] = ptr_u1;

    for (int i = 1; i <= offset[0].count_elem; i++) {
        for (int j = 1; j <= offset[1].count_elem; j++) {
            for (int k = 1; k <= offset[2].count_elem; k++) {
                u0[i][j][k] = an_func((i + offset[0].start_elem - 1) * hx, (j + offset[1].start_elem - 1) * hy, (k + offset[2].start_elem - 1) * hz, 0, ax, ay, az, a_t);
                u1[i][j][k] = u0[i][j][k];
            }
        }
    }
    
    next_iter(n, comm_cart, xy_side, xz_side, yz_side, offset, u1, u0, hx * hx, hy * hy, hz * hz, sqathau / 2);

    double (*arr)[offset[1].count_elem + 2][offset[2].count_elem + 2] = u0;
    double (*cur)[offset[1].count_elem + 2][offset[2].count_elem + 2] = u1;
    double (*tmp)[offset[1].count_elem + 2][offset[2].count_elem + 2];

    double start_time = 0, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    for (int i = 2; i <= steps; i++) {
        next_iter(n, comm_cart, xy_side, xz_side, yz_side, offset, arr, cur, hx * hx, hy * hy, hz * hz, sqathau);
        tmp = cur;
        cur = arr;
        arr = tmp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end_time = MPI_Wtime();

        printf("Max deviation u%d: %.12lf\n", steps, check_layer(offset, cur, ax, ay, az, a_t, hx, hy, hz, time));
        printf("Time spent: %0.6lf\n\n\n", end_time - start_time);

    }

    MPI_Type_free(&y_row);
    MPI_Type_free(&xy_side);
    MPI_Type_free(&xz_side);
    MPI_Type_free(&yz_side);

    free(ptr_u0);
    free(ptr_u1);

    MPI_Finalize();
}
