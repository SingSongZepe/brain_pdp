#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char** argv)
{
  int size, rank;
  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  printf("Hello from process %d of %d\n", rank, size);

  MPI_Finalize();

  for (rank = 0; rank < size; ++rank)
  {
    printf("%d\n", rank);
  }

  return 0;
}