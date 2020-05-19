#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int n, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &n);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int size = 10;
    int *buf_send = new int[size];
    int *buf_recv = new int[size];

    if (rank == 0)
        for (int i = 0; i < size; i++)
            buf_send[i] = i;

    MPI_Allgather(&buf_send, size, MPI_INT, buf_recv, size, comm);

        if (rank == 0)
    {
        printf("%d\n", rank);
    }

    MPI_Finalize();

    return 0;
}