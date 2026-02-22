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
            
            if (value < 0) {
                if (size > 1) {
                    MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
                }
                break;
            }
            
            value += rank;
            printf("Process %d, value = %d\n", rank, value);
            
            if (size > 1) {
                MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            }
        } else {
            int prev_process = rank - 1;
            MPI_Recv(&value, 1, MPI_INT, prev_process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if (value < 0) {
                if (rank < size - 1) {
                    MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
                }
                break;
            }
            
            value += rank;
            printf("Process %d, value = %d\n", rank, value);
            
            if (rank < size - 1) {
                MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}