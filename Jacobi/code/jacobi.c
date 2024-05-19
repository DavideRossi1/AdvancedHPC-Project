#include <mpi.h>
#include <omp.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include "include/printUtilities.h"

/*** function declarations ***/

// save matrix to file
void save_gnuplot( double *M, size_t dim );

// evolve Jacobi
void evolve( double * matrix, double *matrix_new, size_t nRows, size_t nCols, int myRank, int NPEs, MPI_Request* req, MPI_Status* status, int it);

// return the elapsed time
double seconds( void );

// initialize matrix
void init(double* matrix, double* matrix_new, size_t nRows, size_t nCols, int myRank, int NPEs, MPI_Request* req, MPI_Status* status);

/*** end function declaration ***/


int main(int argc, char* argv[])
{  
  // check on input parameters
  if(argc != 3) {
    fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it\n");
    return 1;
  }
  size_t dim = atoi(argv[1]);
  size_t iterations = atoi(argv[2]);
  size_t real_dim = dim + 2;

  int myRank, NPEs,provided;
  MPI_Init_thread(&argc, &argv,MPI_THREAD_MULTIPLE,&provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &NPEs);  

  const uint workSize = real_dim/NPEs;
  const uint workSizeRemainder = real_dim % NPEs;
  const uint myWorkSize = workSize + ((uint)myRank < workSizeRemainder ? 3 : 2);  // 2 rows added for the borders

  size_t my_byte_dim = sizeof(double) * myWorkSize * real_dim;
  double *matrix     = ( double* )malloc( my_byte_dim ); 
  double *matrix_new = ( double* )malloc( my_byte_dim );
  double *tmp_matrix;
  double t_start;
  
  // initialize matrix
  t_start = seconds();
  memset( matrix, 0, my_byte_dim );
  memset( matrix_new, 0, my_byte_dim );
  MPI_Status status;
  MPI_Request req;
  init( matrix, matrix_new, myWorkSize, real_dim, myRank, NPEs, &req, &status);
  
  MPI_Barrier(MPI_COMM_WORLD);
  printMatrixDistributed(matrix, myWorkSize, real_dim, myRank, NPEs);
  MPI_Barrier(MPI_COMM_WORLD);
  if(!myRank) printf("\n");
  printMatrixDistributed(matrix_new, myWorkSize, real_dim, myRank, NPEs);
  MPI_Barrier(MPI_COMM_WORLD);
  if(!myRank) printf("------------------\n");
  MPI_Barrier(MPI_COMM_WORLD);
  printf( "%f\n", seconds() - t_start );
  // start algorithm
  t_start = seconds();
  // for(size_t it = 0; it < iterations; ++it ){
  //   printf("Iteration %zu\n", it);
  //   evolve( matrix, matrix_new, myWorkSize, real_dim, myRank, NPEs, &req, &status, it);
  //   tmp_matrix = matrix;
  //   matrix = matrix_new;
  //   matrix_new = tmp_matrix;
  //  }
  //  printf( "%f\n", seconds() - t_start );

  // t_start = seconds();
  // save_gnuplot( matrix, real_dim );
  // printf( "%f\n", seconds() - t_start );

  free( matrix );
  free( matrix_new );
  MPI_Finalize();
  return 0;
}



void init(double* matrix, double* matrix_new, size_t nRows, size_t nCols, int myRank, int NPEs, MPI_Request* req, MPI_Status* status)
{
  //fill initial values  
  size_t startIndex = myRank ? 1 : 2;
  // fill the inner square
#pragma omp parallel for collapse(2)
  for(size_t i = startIndex; i < nRows-1; ++i )
    for(size_t j = 1; j < nCols-1; ++j )
      matrix[ i*nCols + j ] = 0.5;   
   
  // set up borders 
  uint shift = myRank*(nCols/NPEs) + (myRank < nCols%NPEs ? myRank : nCols%NPEs);
  double increment = 100.0/(nCols-1);
  // fill the first column
#pragma omp parallel for
  for(size_t i = 0; i < nRows-2; ++i ){
    matrix[ (i+1)*nCols ] = (i+shift)*increment;
    matrix_new[ (i+1)*nCols ] = (i+shift)*increment;
  }
  // fill the last row
  if (myRank == NPEs-1){
#pragma omp parallel for
    for(size_t i = 1; i < nCols-1; ++i ){
      matrix[ (nRows-1)*nCols-1-i ] = i*increment;
      matrix_new[ (nRows-1)*nCols-1-i ] = i*increment;
    }
  } else{
#pragma omp parallel for
    for(size_t i = 1; i < nCols-1; ++i )
      matrix[ (nRows-1)*nCols-1-i ] = 0.5;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (NPEs == 1) return;
  #pragma omp master
  {
    if (myRank != 0){
      MPI_Isend(&matrix[nCols], nCols, MPI_DOUBLE, myRank-1, 1, MPI_COMM_WORLD, req);
      MPI_Recv(&matrix[0], nCols, MPI_DOUBLE, myRank-1, 0, MPI_COMM_WORLD, status);

      MPI_Isend(&matrix_new[nCols], nCols, MPI_DOUBLE, myRank-1, 2, MPI_COMM_WORLD, req);
      MPI_Recv(&matrix_new[0], nCols, MPI_DOUBLE, myRank-1, 3, MPI_COMM_WORLD, status);
    }
    if (myRank != NPEs-1){
      MPI_Isend(&matrix[(nRows-2)*nCols], nCols, MPI_DOUBLE, myRank+1, 0, MPI_COMM_WORLD, req);
      MPI_Recv(&matrix[(nRows-1)*nCols], nCols, MPI_DOUBLE, myRank+1, 1, MPI_COMM_WORLD, status);

      MPI_Isend(&matrix_new[(nRows-2)*nCols], nCols, MPI_DOUBLE, myRank+1, 3, MPI_COMM_WORLD, req);
      MPI_Recv(&matrix_new[(nRows-1)*nCols], nCols, MPI_DOUBLE, myRank+1, 2, MPI_COMM_WORLD, status);
    }

    // if (myRank == NPEs -1){
    //   MPI_Isend(&matrix[nCols], nCols, MPI_DOUBLE, myRank-1, 1, MPI_COMM_WORLD, req);
    //   MPI_Recv( &matrix[0],     nCols, MPI_DOUBLE, myRank-1, 0, MPI_COMM_WORLD, status);

    //   MPI_Isend(&matrix_new[nCols], nCols, MPI_DOUBLE, myRank-1, 2, MPI_COMM_WORLD, req);
    //   MPI_Recv( &matrix_new[0],     nCols, MPI_DOUBLE, myRank-1, 3, MPI_COMM_WORLD, status);
    // }
    // if (myRank == 0){
    //   MPI_Isend(&matrix[(nRows-2)*nCols], nCols, MPI_DOUBLE, myRank+1, 0, MPI_COMM_WORLD, req);
    //   MPI_Recv( &matrix[(nRows-1)*nCols], nCols, MPI_DOUBLE, myRank+1, 1, MPI_COMM_WORLD, status);

    //   MPI_Isend(&matrix_new[(nRows-2)*nCols], nCols, MPI_DOUBLE, myRank+1, 3, MPI_COMM_WORLD, req);
    //   MPI_Recv( &matrix_new[(nRows-1)*nCols], nCols, MPI_DOUBLE, myRank+1, 2, MPI_COMM_WORLD, status);
    // }
    // if (myRank != 0 && myRank != NPEs-1){
    //   MPI_Isend(&matrix[(nRows-2)*nCols], nCols, MPI_DOUBLE, myRank+1, 0, MPI_COMM_WORLD, req);
    //   MPI_Isend(&matrix[nCols],           nCols, MPI_DOUBLE, myRank-1, 1, MPI_COMM_WORLD, req);
    //   MPI_Recv( &matrix[(nRows-1)*nCols], nCols, MPI_DOUBLE, myRank+1, 1, MPI_COMM_WORLD, status);
    //   MPI_Recv( &matrix[0],               nCols, MPI_DOUBLE, myRank-1, 0, MPI_COMM_WORLD, status);
      
    //   MPI_Isend(&matrix_new[(nRows-2)*nCols], nCols, MPI_DOUBLE, myRank+1, 3, MPI_COMM_WORLD, req);
    //   MPI_Isend(&matrix_new[nCols],           nCols, MPI_DOUBLE, myRank-1, 2, MPI_COMM_WORLD, req);
    //   MPI_Recv( &matrix_new[(nRows-1)*nCols], nCols, MPI_DOUBLE, myRank+1, 2, MPI_COMM_WORLD, status);
    //   MPI_Recv( &matrix_new[0],               nCols, MPI_DOUBLE, myRank-1, 3, MPI_COMM_WORLD, status);
    // }
  }
}

void evolve( double* matrix, double* matrix_new, size_t nRows, size_t nCols, int myRank, int NPEs, MPI_Request* req, MPI_Status* status, int it)
{
  //This will be a row dominant program.
  for(size_t i = 1 ; i < nRows-1; ++i )
    for(size_t j = 1; j < nCols-1; ++j )
      matrix_new[ i*nCols + j ] = 0.25 * 
        ( matrix[ (i-1)*nCols + j ] + matrix[ i*nCols + (j+1) ] + 	  
          matrix[ (i+1)*nCols + j ] + matrix[ i*nCols + (j-1) ] ); 
  MPI_Barrier(MPI_COMM_WORLD);
  printf("Rank %d, it %d\n", myRank, it);
  printMatrixDistributed(matrix_new, nRows, nCols, myRank, NPEs);
  MPI_Barrier(MPI_COMM_WORLD);

  #pragma omp master
  {
    if (myRank == NPEs-1){
      printf("Sending to %d from %d\n", 0, myRank);
      MPI_Isend(&matrix_new[(nRows-2)*nCols], nCols, MPI_DOUBLE, 0, 2*it+1, MPI_COMM_WORLD, req);
      MPI_Isend(&matrix_new[nCols],           nCols, MPI_DOUBLE, myRank-1, 2*it,   MPI_COMM_WORLD, req);
      MPI_Recv(&matrix_new,                  nCols, MPI_DOUBLE, myRank-1, 2*it,   MPI_COMM_WORLD, status);
      MPI_Recv(&matrix_new[(nRows-1)*nCols], nCols, MPI_DOUBLE, 0, 2*it+1, MPI_COMM_WORLD, status);
    }
    if (myRank == 0){
      printf("Sending to %d from %d\n", NPEs-1, myRank);
      MPI_Isend(&matrix_new[(nRows-2)*nCols], nCols, MPI_DOUBLE, myRank+1, 2*it,   MPI_COMM_WORLD, req);
      MPI_Isend(&matrix_new[nCols],           nCols, MPI_DOUBLE, NPEs-1, 2*it+1, MPI_COMM_WORLD, req);
      MPI_Recv(&matrix_new,                  nCols, MPI_DOUBLE, NPEs-1, 2*it,   MPI_COMM_WORLD, status);
      MPI_Recv(&matrix_new[(nRows-1)*nCols], nCols, MPI_DOUBLE, myRank+1, 2*it+1, MPI_COMM_WORLD, status);
    }
    if (myRank !=0 && myRank != NPEs-1){
      printf("Sending to %d from %d\n", myRank-1, myRank);
      MPI_Isend(&matrix_new[(nRows-2)*nCols], nCols, MPI_DOUBLE, myRank+1, 2*it,   MPI_COMM_WORLD, req);
      MPI_Isend(&matrix_new[nCols],           nCols, MPI_DOUBLE, myRank-1, 2*it+1, MPI_COMM_WORLD, req);
      MPI_Recv(&matrix_new,                  nCols, MPI_DOUBLE, myRank-1, 2*it,   MPI_COMM_WORLD, status);
      MPI_Recv(&matrix_new[(nRows-1)*nCols], nCols, MPI_DOUBLE, myRank+1, 2*it+1, MPI_COMM_WORLD, status);
    }
    // MPI_Isend(&matrix[(nRows-2)*nCols], nCols, MPI_DOUBLE, (myRank+1)%NPEs, 2*it,   MPI_COMM_WORLD, req);
    // MPI_Isend(&matrix[nCols],           nCols, MPI_DOUBLE, (myRank-1)%NPEs, 2*it+1, MPI_COMM_WORLD, req);

    // MPI_Recv(&matrix,                  nCols, MPI_DOUBLE, (myRank-1)%NPEs, 2*it,   MPI_COMM_WORLD, status);
    // MPI_Recv(&matrix[(nRows-1)*nCols], nCols, MPI_DOUBLE, (myRank+1)%NPEs, 2*it+1, MPI_COMM_WORLD, status);
  }
}

void save_gnuplot( double *M, size_t dim )
{
  const double h = 0.1;
  FILE *file = fopen( "solution.dat", "w" );
  for(size_t i = 0; i < dim; ++i )
    for(size_t j = 0; j < dim; ++j )
      fprintf(file, "%f\t%f\t%f\n", h*j, -h*i, M[i*dim + j] );
  fclose( file );
}



// A Simple timer for measuring the walltime
double seconds()
{
    struct timeval tmp;
    double sec;
    gettimeofday( &tmp, (struct timezone *)0 );
    sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
    return sec;
}
