#include <stdio.h>
#include <string.h>
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
    uint charRowSize = getRowSize(nCols);
    size_t entireCharMatrixSize = nCols * charRowSize + 1; // the entire matrix is nCols*nCols big, +1 for the null terminator
    char *entireCharMatrix = (char*)malloc(entireCharMatrixSize * sizeof(char));
    char* charMatrix = (char*)malloc(nRows * charRowSize * sizeof(char));
    convertMatrix(matrix, charMatrix, nRows, nCols);
    offset += snprintf(entireCharMatrix + offset, entireCharMatrixSize - offset, "%s", charMatrix);
    free(charMatrix);
    for (uint i = 1; i < NPEs; i++) 
    {
      uint nRowsSender = nCols / NPEs + (i < nCols % NPEs ? 1 : 0);
      charMatrix = (char*)malloc((nRowsSender * charRowSize + 1) * sizeof(char)); // +1 for the null terminator
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
  const size_t charMatrixSize = nRows*getRowSize(nCols) + 1; 
  for (uint i = 0; i < nRows; i++) {  
    for (uint j = 0; j < nCols; j++){
      char* format = j < nCols-1 ? "%.3f\t" : "%.3f\n";
      offset += snprintf(charMatrix+offset, charMatrixSize-offset, format, matrix[i * nCols + j]);
    }
  }
}

size_t getRowSize(uint matrixSize) {
  size_t maxVal = matrixSize*matrixSize;
  // Explanation: 
  // - if the matrix has more than 100 elements, the maximum value will have 3 digits, so we need 8 characters to represent it (123.456\t)
  // - if the matrix has more than 10 elements, the maximum value will have 2 digits, so we need 7 characters to represent it (12.345\t)
  // - if the matrix has 10 or less elements, the maximum value will have 1 digit, so we need 6 characters to represent it (1.234\t)
  // This is done just to save some memory, we could have simply used 8 characters for all cases, or even more to cover any kind of matrix,
  // but that would be a waste of memory.
  int elemSize = maxVal > 100 ? 8 : maxVal > 10 ? 7 : 6;
  return matrixSize * elemSize;
}

void printMatrixDistributed(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs) 
{
  #pragma acc update self(matrix[:nRows*nCols])
  if (myRank) {
    if (myRank == NPEs-1) nRows++;
    MPI_Send(&matrix[nCols], (nRows-2) * nCols, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  } else 
  {
    uint nRowsCurrentProc = nRows - (NPEs > 1 ? 1 : 0);
    printMatrix(matrix, nRowsCurrentProc, nCols);
    for (uint i = 1; i < NPEs; i++) 
    {
      nRowsCurrentProc = (nCols-2)/NPEs + (i < (nCols-2)%NPEs ? 1 : 0) + (i == NPEs-1 ? 1 : 0);
      double *buf = (double *)malloc(nRowsCurrentProc * nCols * sizeof(double));
      MPI_Recv(buf, nRowsCurrentProc * nCols, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printMatrix(buf, nRowsCurrentProc, nCols);
      free(buf);
    }
    printf("\n");
  }
}

void printMatrix(double *matrix, uint nRows, uint nCols) {
  for (uint i = 0; i < nRows; i++) {
    for (uint j = 0; j < nCols; j++)
      printf("%.3f\t", matrix[i*nCols + j]);
    printf("\n");
  }
}

void save_gnuplot( double *M, size_t dim, uint firstRow, uint lastRow, uint shift)
{
  const double h = 0.1;
  MPI_File file;
  MPI_File_open(MPI_COMM_WORLD, "solution.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
  MPI_Offset of = shift*dim*3*sizeof(double);
  for (size_t i = firstRow; i < lastRow; i++){
    for (size_t j = 0; j < dim; j++){
      double buffer[3] = {h*j, -h*(shift+i), M[i*dim+j]};
      MPI_File_write_at(file, of + (i*dim+j)*3*sizeof(double), buffer, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
  }
  MPI_File_close(&file);
}
