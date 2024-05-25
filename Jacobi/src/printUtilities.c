#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "printUtilities.h"

void convertMatrix(double *matrix, char* buffer, uint nRows, uint nCols) {
  const char* format = "%.3f\t";
  // build a string that contains the matrix
  int offset = 0;
  const size_t bufferSize = nRows*(nCols*7 + 2);
  for (uint i = 0; i < nRows; i++) {  // skip the first and last rows, which are used to exchange messages with the other procs
    for (uint j = 0; j < nCols; j++){
      offset += snprintf(buffer+offset, bufferSize-offset, format, matrix[i * nCols + j]);
    }
    offset += snprintf(buffer+offset, bufferSize-offset, "\n");
  }
}

void printMatrixDistributed(double *matrix, uint nRows, uint nCols, uint myRank, uint NPEs) 
{
  if (myRank) {
    if (myRank == NPEs-1) nRows++;
    MPI_Send(&matrix[nCols], nRows*nCols, MPI_DOUBLE, 0, myRank, MPI_COMM_WORLD);
  } 
  else {
    int offset = 0;
    size_t rowCharSize = nCols*7 + 2; // 7 is the number of characters in the format string ('90.000\t'), 1 is for the newline
    // Note: we actually also have a 100.000\t, but it will be a single value in the matrix, so the space needed for it is balanced
    // by the zeroes in the first row
    int nRowsPE1 = nRows + (NPEs == 1 ? 2 : 1);
    char* myPart =       (char*)malloc(nRowsPE1 * rowCharSize * sizeof(char));
    char *entireMatrix = (char*)malloc(nCols * rowCharSize * sizeof(char)); // the entire matrix is nCols*nCols big
    convertMatrix(matrix, myPart, nRowsPE1, nCols);
    offset += snprintf(entireMatrix + offset, nRowsPE1 * rowCharSize - offset, "%s", myPart);
    double *buf = (double *)malloc((nRows+1) * nCols * sizeof(double)); // +1 because the last PE has an extra row
    for (uint i = 1; i < NPEs; i++) {
      uint nRowsSender = (nCols-2)/NPEs + (i < (nCols-2)%NPEs ? 1 : 0) + (i == NPEs-1 ? 1 : 0);
      MPI_Recv(buf, nRowsSender * nCols, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      convertMatrix(buf, myPart, nRowsSender, nCols);
      offset += snprintf(entireMatrix+offset, nRowsSender*rowCharSize-offset, "%s", myPart);
    }
    printf("%s", entireMatrix);
    free(buf);
    free(myPart);
    free(entireMatrix);
  }
}

void save_gnuplot( double *M, size_t dim, uint myRank, uint NPEs, uint myWorkSize)
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
      MPI_File_write_at(file, of + (i*dim+j)*3*sizeof(double), buffer, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
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