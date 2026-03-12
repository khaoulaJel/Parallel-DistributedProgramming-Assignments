#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int NX = 20, NY = 20, G = 20;
    if (argc > 1) NX = atoi(argv[1]);
    if (argc > 2) NY = atoi(argv[2]);
    if (argc > 3) G = atoi(argv[3]);

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int Px = dims[0], Py = dims[1];

    int periods[2] = {1, 1}; 
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int north, south, west, east;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);

    int local_nx = NX / Px;
    int local_ny = NY / Py;

    int *grid_data  = (int*)calloc((local_nx+2)*(local_ny+2), sizeof(int));
    int *nextg_data = (int*)calloc((local_nx+2)*(local_ny+2), sizeof(int));

    int **grid  = (int**)malloc((local_nx+2) * sizeof(int*));
    int **nextg = (int**)malloc((local_nx+2) * sizeof(int*));
    for (int i = 0; i < local_nx+2; i++) {
        grid[i]  = grid_data + i*(local_ny+2);
        nextg[i] = nextg_data + i*(local_ny+2);
    }

    for (int i = 1; i <= local_nx; i++) {
        int gx = coords[0]*local_nx + i-1;
        for (int j = 1; j <= local_ny; j++) {
            int gy = coords[1]*local_ny + j-1;
            grid[i][j] = ((gx + gy) % 3 == 1) ? 1 : 0;
        }
    }

    MPI_Datatype column_type;
    MPI_Type_vector(local_nx, 1, local_ny+2, MPI_INT, &column_type);
    MPI_Type_commit(&column_type);

    MPI_Status status;

    for (int gen = 0; gen < G; gen++) {

        MPI_Sendrecv(&grid[1][1], local_ny, MPI_INT, north, 0,
                     &grid[local_nx+1][1], local_ny, MPI_INT, south, 0,
                     cart_comm, &status);

        MPI_Sendrecv(&grid[local_nx][1], local_ny, MPI_INT, south, 1,
                     &grid[0][1], local_ny, MPI_INT, north, 1,
                     cart_comm, &status);

        MPI_Sendrecv(&grid[1][1], 1, column_type, west, 2,
                     &grid[1][local_ny+1], 1, column_type, east, 2,
                     cart_comm, &status);

        MPI_Sendrecv(&grid[1][local_ny], 1, column_type, east, 3,
                     &grid[1][0], 1, column_type, west, 3,
                     cart_comm, &status);

        for (int i = 1; i <= local_nx; i++) {
            for (int j = 1; j <= local_ny; j++) {
                int live = grid[i-1][j-1] + grid[i-1][j] + grid[i-1][j+1] +
                           grid[i][j-1]             + grid[i][j+1] +
                           grid[i+1][j-1] + grid[i+1][j] + grid[i+1][j+1];
                nextg[i][j] = (grid[i][j] == 1) ? (live == 2 || live == 3) : (live == 3);
            }
        }

        for (int i = 1; i <= local_nx; i++)
            for (int j = 1; j <= local_ny; j++)
                grid[i][j] = nextg[i][j];

        if (gen % 5 == 0 || gen == G-1) {
            MPI_Barrier(cart_comm);
            for (int r = 0; r < size; r++) {
                if (rank == r) {
                    printf("Rank %d (coords %d,%d) - Generation %d:\n", rank, coords[0], coords[1], gen);
                    for (int i = 1; i <= local_nx; i++) {
                        for (int j = 1; j <= local_ny; j++)
                            printf("%d ", grid[i][j]);
                        printf("\n");
                    }
                    printf("\n");
                }
                MPI_Barrier(cart_comm);
            }
        }
    }

    MPI_Type_free(&column_type);
    free(grid);
    free(nextg);
    free(grid_data);
    free(nextg_data);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}