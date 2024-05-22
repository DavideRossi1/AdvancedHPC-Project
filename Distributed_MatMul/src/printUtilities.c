#include <stdio.h>
#include <mpi.h>

#include "printUtilities.h"

void printMatrix(double *matrix, uint nRows, uint nCols) {
  #ifdef DEBUG
    const char* format = "%.0f ";
  #else
    const char* format = "%.2f ";
  #endif
  for (uint i = 0; i < nRows; i++) {
    for (uint j = 0; j < nCols; j++)
      printf(format, matrix[i * nCols + j]);
    printf("\n");
  }
}

void printMatrixDistributed(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs) 
{
  if (myRank) {
    MPI_Send(matrix, nRows * nCols, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  } 
  else {
    double *buf = (double *)malloc(nRows * nCols * sizeof(double));
    printMatrix(matrix, nRows, nCols);
    for (uint i = 1; i < NPEs; i++) {
      uint nLocSender = nCols / NPEs + (i < nCols % NPEs ? 1 : 0);
      MPI_Recv(buf, nLocSender * nCols, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printMatrix(buf, nLocSender, nCols);
    }
    free(buf);
  }
}
