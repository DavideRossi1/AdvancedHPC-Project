#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "../include/printUtilities.h"

void printMatrix(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs) {
  const char* format = "%.2f ";
  size_t bufferSize = nRows*(nCols*7 + 2); // 7 is the number of characters in the format string ('100.00 '), 2 is for the newline and null terminator
  char* buffer = (char*)malloc(bufferSize*sizeof(char));
  // build a string that contains the matrix
  int offset = 0;
  uint start = myRank == 0 ? 0 : 1;
  uint end = myRank == NPEs-1 ? nRows : nRows-1;
  for (uint i = start; i < end; i++) {  // skip the first and last rows, which are used to exchange messages with the other procs
    for (uint j = 0; j < nCols; j++){
      offset += snprintf(buffer+offset, bufferSize-offset, format, matrix[i * nCols + j]);
    }
    offset += snprintf(buffer+offset, bufferSize-offset, "\n");
  }
  printf("%s", buffer);
  free(buffer);
}

void printMatrixDistributed(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs) 
{
  if (myRank) {
    MPI_Send(matrix, nRows * nCols, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  } 
  else {
    double *buf = (double *)malloc(nRows * nCols * sizeof(double));
    printMatrix(matrix, nRows, nCols, myRank, NPEs);
    #ifdef DEBUG
      printf("done printing rank %d\n", myRank);
    #endif
    for (uint i = 1; i < NPEs; i++) {
      uint nLocSender = (nCols-2)/NPEs + (i < (nCols-2)%NPEs ? 3 : 2);
      MPI_Recv(buf, nLocSender * nCols, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printMatrix(buf, nLocSender, nCols, i, NPEs);
      #ifdef DEBUG
        printf("done printing rank %d\n", i);
      #endif
    }
    free(buf);
  }
}

void save_gnuplot( double *M, size_t dim, uint myRank, uint NPEs, uint myWorkSize, MPI_Status status)
{
  const double h = 0.1;
  MPI_File file;
  MPI_File_open(MPI_COMM_WORLD, "solution.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
  uint shift = myRank*((dim-2)/NPEs) + (myRank < (dim-2)%NPEs ? myRank : (dim-2)%NPEs);
  MPI_Offset of = shift*dim*3*sizeof(double);
  size_t start = myRank ? 1 : 0;
  size_t end = myRank==NPEs-1 ? myWorkSize : myWorkSize-1;
  for (size_t i = start; i < end; i++){
    for (size_t j = 0; j < dim; j++){
      double buffer[3] = {h*j, -h*i, M[i*dim+j]};
      MPI_File_write_at(file, of + (i*dim+j)*3*sizeof(double), buffer, 3, MPI_DOUBLE, &status);
    }
  }
  MPI_File_close(&file);
}


void convertBinToTxt()
{
    FILE *binFile, *txtFile;
    double buffer[3];
    binFile = fopen("solution.dat", "rb");
    if (binFile == NULL) {
        fprintf(stderr, "Cannot open binary file.\n");
        return;
    }
    txtFile = fopen("solution.csv", "w");
    if (txtFile == NULL) {
        fprintf(stderr, "Cannot open text file.\n");
        fclose(binFile);
        return;
    }
    // Read from binary file and write to text file
    while (fread(buffer, sizeof(double), 3, binFile) == 3)
        fprintf(txtFile, "%.6f\t%.6f\t%.6f\n", buffer[0], buffer[1], buffer[2]);
    fclose(binFile);
    fclose(txtFile);

}