/*
 * Process 0 sends a 4x5 matrix to Process 1.
 * Process 1 receives it already transposed (5x4) using a derived datatype.
 */

#include <mpi.h>
#include <stdio.h>

#define ROWS 4
#define COLS 5

int main(int argc, char *argv[]) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        int a[ROWS][COLS];
        int val = 1;
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                a[i][j] = val++;

        printf("Process 0 - Matrix a:\n");
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++)
                printf("%d\t", a[i][j]);
            printf("\n");
        }

        MPI_Send(&a[0][0], ROWS * COLS, MPI_INT, 1, 0, MPI_COMM_WORLD);

    } else if (rank == 1) {
        int at[COLS][ROWS];   

        /*
         * Build a derived type that maps the incoming row-major stream of A
         * directly into the transposed layout of AT.
         *
         * Pattern observed:
         *   incoming element i  ->  memory position (i%COLS)*ROWS + (i/COLS)
         *
         * This groups into ROWS groups of COLS elements each, where each
         * group is placed with stride ROWS in memory, and groups start
         * 1 double apart.
         *
         * Step 1 : MPI_Type_vector
         *   - COLS  : number of elements per group (one full row of AT)
         *   - 1     : blocklen (one element at a time)
         *   - ROWS  : stride in units of MPI_INT (distance between elements
         *             in the same group inside AT's memory)
         */
        MPI_Datatype row_type, transpose_type;

        MPI_Type_vector(COLS, 1, ROWS, MPI_INT, &row_type);

        /*
         * Step 2 : MPI_Type_create_hvector
         *   - ROWS          : number of groups (= number of rows of A)
         *   - 1             : one row_type per group
         *   - sizeof(int)   : stride in BYTES between group starting addresses
         *                     (each group starts exactly 1 int later in AT)
         */
        MPI_Type_create_hvector(ROWS, 1, sizeof(int), row_type, &transpose_type);
        MPI_Type_commit(&transpose_type);

        /* Receive with count=1 of our custom type */
        MPI_Recv(&at[0][0], 1, transpose_type, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        printf("Process 1 - Matrix transposee at:\n");
        for (int i = 0; i < COLS; i++) {
            for (int j = 0; j < ROWS; j++)
                printf("%d\t", at[i][j]);
            printf("\n");
        }

        MPI_Type_free(&row_type);
        MPI_Type_free(&transpose_type);
    }

    MPI_Finalize();
    return 0;
}