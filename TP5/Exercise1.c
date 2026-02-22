#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]){

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    

    printf("Hello World\n");

    printf("Process %d out of %d processes\n", rank, size);

    if (rank == 0) {
        printf("I am the master process\n");
    }
    MPI_Finalize();
    return 0 ;

}