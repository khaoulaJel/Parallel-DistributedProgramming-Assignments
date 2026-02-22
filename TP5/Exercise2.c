#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int value;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    while (1) {
        if (rank == 0) {
            printf("Enter an integer (negative to exit): ");
            scanf("%d", &value);
        }
        
        MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (value < 0) {
            break;
        }
        
        printf("Process %d got %d\n", rank, value);
    }
    
    MPI_Finalize();
    return 0;
}