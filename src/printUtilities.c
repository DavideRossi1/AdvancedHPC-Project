#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#include "../include/printUtilities.h"

void printMatrix(double *matrix, int nRows, int nCols) {
  for (int i = 0; i < nRows; i++) {
    for (int j = 0; j < nCols; j++)
      printf("%.0f ", matrix[i * nCols + j]);
    printf("\n");
  }
}

void printMatrixDistributed(double *matrix, int nRows, int nCols,
                                   int myPID, int NPEs) {
  if (myPID) {
    MPI_Send(matrix, nRows * nCols, MPI_DOUBLE, 0, myPID, MPI_COMM_WORLD);
  } else {
    double *buf = (double *)malloc(nRows * nCols * sizeof(double));
    printMatrix(matrix, nRows, nCols);
    for (int i = 1; i < NPEs; i++) {
      int nLocSender = nCols / NPEs + (i < nCols % NPEs ? 1 : 0);
      MPI_Recv(buf, nLocSender * nCols, MPI_DOUBLE, i, i, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      printMatrix(buf, nLocSender, nCols);
    }
    free(buf);
  }
}
