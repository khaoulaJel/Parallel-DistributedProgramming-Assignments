#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int ntx, nty, sx, ex, sy, ey;
double *f;
double coef[3];

#define IDX(i, j) (((i)-(sx-1)) * (ey - sy + 3) + (j) - (sy - 1))

void initialization(double **pu, double **pu_new, double **pu_exact) {
    double hx = 1.0 / (ntx + 1.0);
    double hy = 1.0 / (nty + 1.0);

    *pu = calloc((ex - sx + 3) * (ey - sy + 3), sizeof(double));
    *pu_new = calloc((ex - sx + 3) * (ey - sy + 3), sizeof(double));
    *pu_exact = calloc((ex - sx + 3) * (ey - sy + 3), sizeof(double));
    f = calloc((ex - sx + 3) * (ey - sy + 3), sizeof(double));

    coef[0] = (0.5 * hx * hx * hy * hy) / (hx * hx + hy * hy);
    coef[1] = 1.0 / (hx * hx);
    coef[2] = 1.0 / (hy * hy);

    for (int iterx = sx; iterx <= ex; iterx++) {
        for (int itery = sy; itery <= ey; itery++) {
            double x = iterx * hx;
            double y = itery * hy;
            f[IDX(iterx, itery)] = 2.0 * (x * x - x + y * y - y);
            (*pu_exact)[IDX(iterx, itery)] = x * y * (x - 1.0) * (y - 1.0);
        }
    }
}

void compute(double *u, double *u_new) {
    for (int i = sx; i <= ex; i++) {
        for (int j = sy; j <= ey; j++) {
            u_new[IDX(i, j)] =
                coef[0] * (
                    coef[1] * (u[IDX(i + 1, j)] + u[IDX(i - 1, j)]) +
                    coef[2] * (u[IDX(i, j + 1)] + u[IDX(i, j - 1)]) -
                    f[IDX(i, j)]
                );
        }
    }
}

void output_results(double *u, double *u_exact) {
    printf("Exact solution uexact - Computed solution u\n");
    for (int j = sy; j <= ey; j++)
        printf("%12.5e - %12.5e\n", u_exact[IDX(1, j)], u[IDX(1, j)]);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ntx = 12; nty = 10;
    if (argc > 1) ntx = atoi(argv[1]);
    if (argc > 2) nty = atoi(argv[2]);

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int north, south, west, east;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);

    int local_nx = ntx / dims[0];
    int local_ny = nty / dims[1];
    sx = 1 + coords[0] * local_nx;
    ex = sx + local_nx - 1;
    sy = 1 + coords[1] * local_ny;
    ey = sy + local_ny - 1;

    if (rank == 0) {
        printf("Poisson execution with %d MPI processes\n", size);
        printf("Domain size: ntx=%d nty=%d\n", ntx, nty);
        printf("Topology dimensions: %d along x, %d along y\n\n", dims[0], dims[1]);
    }

    MPI_Barrier(cart_comm);

    for (int r = 0; r < size; r++) {
        if (rank == r) {
            printf("Rank in the topology: %d Array indices: x from %d to %d, y from %d to %d\n",
                   rank, sx, ex, sy, ey);
            printf("Process %d has neighbors: N %d E %d S %d W %d\n",
                   rank, north, east, south, west);
        }
        MPI_Barrier(cart_comm);
    }

    double *u, *u_new, *u_exact;
    initialization(&u, &u_new, &u_exact);

    double start_time = MPI_Wtime();
    int max_iter = 10000;
    double tol = 1.e-15;
    int iter;
    double global_err = 1.0;
    int nx_loc = ex - sx + 1;
    int ny_loc = ey - sy + 1;
    double *col_buf = malloc(nx_loc * sizeof(double));

    for (iter = 0; iter < max_iter; iter++) {
        MPI_Status status;

        if (north != MPI_PROC_NULL) {
            MPI_Sendrecv(&u[IDX(sx, sy)], ny_loc, MPI_DOUBLE, north, 0,
                         &u[IDX(sx-1, sy)], ny_loc, MPI_DOUBLE, north, 1,
                         cart_comm, &status);
        }
        if (south != MPI_PROC_NULL) {
            MPI_Sendrecv(&u[IDX(ex, sy)], ny_loc, MPI_DOUBLE, south, 1,
                         &u[IDX(ex+1, sy)], ny_loc, MPI_DOUBLE, south, 0,
                         cart_comm, &status);
        }
        if (west != MPI_PROC_NULL) {
            for (int k = 0, i = sx; i <= ex; i++, k++) col_buf[k] = u[IDX(i, sy)];
            MPI_Sendrecv(col_buf, nx_loc, MPI_DOUBLE, west, 2,
                         col_buf, nx_loc, MPI_DOUBLE, west, 3,
                         cart_comm, &status);
            for (int k = 0, i = sx; i <= ex; i++, k++) u[IDX(i, sy-1)] = col_buf[k];
        }
        if (east != MPI_PROC_NULL) {
            for (int k = 0, i = sx; i <= ex; i++, k++) col_buf[k] = u[IDX(i, ey)];
            MPI_Sendrecv(col_buf, nx_loc, MPI_DOUBLE, east, 3,
                         col_buf, nx_loc, MPI_DOUBLE, east, 2,
                         cart_comm, &status);
            for (int k = 0, i = sx; i <= ex; i++, k++) u[IDX(i, ey+1)] = col_buf[k];
        }

        compute(u, u_new);

        double local_err = 0.0;
        for (int i = sx; i <= ex; i++) {
            for (int j = sy; j <= ey; j++) {
                double diff = fabs(u_new[IDX(i, j)] - u[IDX(i, j)]);
                if (diff > local_err) local_err = diff;
            }
        }
        MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, cart_comm);

        if (rank == 0 && iter % 100 == 0 && iter > 0) {
            printf("Iteration %d global_error = %.5e\n", iter, global_err);
        }

        double *tmp = u; u = u_new; u_new = tmp;

        if (global_err < tol) break;
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Converged after %d iterations in %.6f seconds\n", iter + 1, end_time - start_time);
        output_results(u, u_exact);
    }

    free(col_buf); free(u); free(u_new); free(u_exact); free(f);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}