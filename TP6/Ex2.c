/*
 * Exercise 2 : Distributed Gradient Descent with MPI Derived Types
 *
 * Linear regression with MSE loss, parallelised via MPI.
 * Uses MPI_Type_create_struct for the Sample type and MPI_Scatterv
 * to distribute data across processes.
 *
 * Compile : mpicc -o ex2_grad_descent ex2_grad_descent.c -lm
 * Run     : mpirun -np 4 ./ex2_grad_descent
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Configuration                                                       */
/* ------------------------------------------------------------------ */
#define N_SAMPLES     500000
#define N_FEATURES    5
#define MAX_EPOCHS    10000
#define LEARNING_RATE 0.01
#define LOSS_THRESHOLD 1e-2
#define PRINT_EVERY   10

/* ------------------------------------------------------------------ */
/*  Data structure                                                      */
/* ------------------------------------------------------------------ */
typedef struct {
    double x[N_FEATURES];
    double y;
} Sample;

/* ------------------------------------------------------------------ */
/*  Generate synthetic data  (only called on process 0)               */
/* ------------------------------------------------------------------ */
void generate_data(Sample *data, int n) {
    double w_true[N_FEATURES] = {2.0, -1.0, 0.5, 1.5, -0.5};
    srand(42);
    for (int i = 0; i < n; i++) {
        double y = 0.0;
        for (int f = 0; f < N_FEATURES; f++) {
            data[i].x[f] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            y += w_true[f] * data[i].x[f];
        }
        data[i].y = y + ((double)rand() / RAND_MAX - 0.5) * 0.3;
    }
}

/* ------------------------------------------------------------------ */
/*  Build MPI derived type for Sample                                  */
/* ------------------------------------------------------------------ */
MPI_Datatype build_sample_type(void) {
    MPI_Datatype sample_type;
    Sample dummy;

    int          blocklengths[2] = {N_FEATURES, 1};
    MPI_Datatype types[2]        = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint     displacements[2];

    MPI_Aint base_addr;
    MPI_Get_address(&dummy,      &base_addr);
    MPI_Get_address(&dummy.x[0], &displacements[0]);
    MPI_Get_address(&dummy.y,    &displacements[1]);

    displacements[0] -= base_addr;
    displacements[1] -= base_addr;

    MPI_Type_create_struct(2, blocklengths, displacements, types, &sample_type);
    MPI_Type_commit(&sample_type);
    return sample_type;
}

/* ------------------------------------------------------------------ */
/*  main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype MPI_SAMPLE = build_sample_type();

    /* How many samples per process */
    int local_n   = N_SAMPLES / size;
    int remainder = N_SAMPLES % size;

    int *sendcounts = malloc(size * sizeof(int));
    int *displs     = malloc(size * sizeof(int));
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = local_n + (i < remainder ? 1 : 0);
        displs[i]     = offset;
        offset       += sendcounts[i];
    }
    int my_count = sendcounts[rank];

    Sample *all_data   = NULL;
    Sample *local_data = malloc(my_count * sizeof(Sample));

    if (rank == 0) {
        all_data = malloc(N_SAMPLES * sizeof(Sample));
        generate_data(all_data, N_SAMPLES);
    }

    /* Scatter samples */
    MPI_Scatterv(all_data,   sendcounts, displs, MPI_SAMPLE,
                 local_data, my_count,           MPI_SAMPLE,
                 0, MPI_COMM_WORLD);

    /* Initialise weights */
    double w[N_FEATURES];
    memset(w, 0, sizeof(w));

    double start_time = MPI_Wtime();
    double global_loss = 1e9;
    int epoch;

    for (epoch = 1; epoch <= MAX_EPOCHS; epoch++) {

        double local_grad[N_FEATURES];
        memset(local_grad, 0, sizeof(local_grad));
        double local_loss = 0.0;

        for (int i = 0; i < my_count; i++) {
            double pred = 0.0;
            for (int f = 0; f < N_FEATURES; f++)
                pred += w[f] * local_data[i].x[f];
            double error = pred - local_data[i].y;
            local_loss += error * error;
            for (int f = 0; f < N_FEATURES; f++)
                local_grad[f] += error * local_data[i].x[f];
        }

        double global_grad[N_FEATURES];
        MPI_Allreduce(local_grad, global_grad, N_FEATURES,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_loss, &global_loss, 1,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_loss /= N_SAMPLES;

        for (int f = 0; f < N_FEATURES; f++)
            w[f] -= (LEARNING_RATE / N_SAMPLES) * global_grad[f];

        if (rank == 0 && epoch % PRINT_EVERY == 0)
            printf("Epoch %4d | Loss (MSE): %.6f | w[0]: %.4f, w[1]: %.4f\n",
                   epoch, global_loss, w[0], w[1]);

        if (global_loss < LOSS_THRESHOLD) {
            if (rank == 0)
                printf("Early stopping at epoch %d - loss %.6f < %.1e\n",
                       epoch, global_loss, LOSS_THRESHOLD);
            break;
        }
    }

    double elapsed = MPI_Wtime() - start_time;
    if (rank == 0) {
        printf("\nTraining time: %.3f seconds (MPI, %d processes)\n",
               elapsed, size);
        printf("Final weights:");
        for (int f = 0; f < N_FEATURES; f++)
            printf("  w[%d]=%.4f", f, w[f]);
        printf("\n");
    }

    MPI_Type_free(&MPI_SAMPLE);
    free(local_data);
    free(sendcounts);
    free(displs);
    if (rank == 0) free(all_data);

    MPI_Finalize();
    return 0;
}