#include <stdio.h>
#include <mpi.h>

#include "print.h"

void printMatrixThrSafe(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs) 
{
  MPI_Barrier(MPI_COMM_WORLD);
  if (myRank) {
    MPI_Send(matrix, nRows * nCols, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  } else 
  {
    size_t offset = 0;
    uint charRowSize = nCols*7; // 7 is the number of characters in a string like '12.345\t', 1 is for the newline character
    // We may overall allocate slightly more memory than needed, but it's ok since we'll only use this function for debugging purposes.
    size_t entireCharMatrixSize = nCols * charRowSize; // the entire matrix is nCols*nCols big
    char *entireCharMatrix = (char*)malloc(entireCharMatrixSize * sizeof(char));
    char* charMatrix = (char*)malloc(nRows * charRowSize * sizeof(char));
    convertMatrix(matrix, charMatrix, nRows, nCols);
    offset += snprintf(entireCharMatrix + offset, entireCharMatrixSize - offset, "%s", charMatrix);
    free(charMatrix);
    for (uint i = 1; i < NPEs; i++) 
    {
      uint nRowsSender = nCols / NPEs + (i < nCols % NPEs ? 1 : 0);
      charMatrix = (char*)malloc(nRowsSender * charRowSize * sizeof(char));
      double *buf = (double *)malloc(nRowsSender * nCols * sizeof(double));
      MPI_Recv(buf, nRowsSender * nCols, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      convertMatrix(buf, charMatrix, nRowsSender, nCols);
      offset += snprintf(entireCharMatrix + offset, entireCharMatrixSize - offset, "%s", charMatrix);
      free(charMatrix);
      free(buf);
    }
    printf("%s\n", entireCharMatrix);
    free(entireCharMatrix);
  }
}

void convertMatrix(double *matrix, char* charMatrix, uint nRows, uint nCols) {
  // build a string that contains the matrix
  size_t offset = 0;
  const size_t charMatrixSize = nRows*(nCols*7);
  for (uint i = 0; i < nRows; i++) {  
    for (uint j = 0; j < nCols; j++){
      char* format = j < nCols-1 ? "%.3f\t" : "%.3f\n";
      // maxlen has a +1 to account for the null terminator, which would overwrite the final \n if
      // all matrix elements have exactly 7 characters
      offset += snprintf(charMatrix+offset, charMatrixSize-offset + 1, format, matrix[i * nCols + j]);
    }
  }
}

void printMatrixDistributed(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs) 
{
  if (myRank) {
    MPI_Send(matrix, nRows * nCols, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  } 
  else {
    printMatrix(matrix, nRows, nCols);
    for (uint i = 1; i < NPEs; i++) {
      uint nRowsSender = nCols / NPEs + (i < nCols % NPEs ? 1 : 0);
      double *buf = (double *)malloc(nRowsSender * nCols * sizeof(double));
      MPI_Recv(buf, nRowsSender * nCols, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printMatrix(buf, nRowsSender, nCols);
      free(buf);
    }
    printf("\n");
  }
}

void printMatrix(double *matrix, uint nRows, uint nCols) {
  for (uint i = 0; i < nRows; i++) {
    for (uint j = 0; j < nCols; j++)
      printf("%.3f\t", matrix[i * nCols + j]);
    printf("\n");
  }
}
