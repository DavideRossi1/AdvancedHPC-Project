#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#include "print.h"
#include "init.h"
#include "evolve.h"
#include "timer.h"

int main(int argc, char* argv[])
{  
  if(argc != 3) {
    fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it\n");
    return 1;
  }
  int myRank, NPEs,provided;
  MPI_Init_thread(&argc, &argv,MPI_THREAD_MULTIPLE,&provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &NPEs);  

  struct Timer t = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  MPI_Barrier(MPI_COMM_WORLD);
  t.programStart = MPI_Wtime();
  start(&t);
  size_t dim = atoi(argv[1]);
  size_t dimWithEdges = dim + 2;
  size_t iterations = atoi(argv[2]);
  int prev = myRank ? myRank-1 : MPI_PROC_NULL;
  int next = myRank != NPEs-1 ? myRank+1 : MPI_PROC_NULL;

  const uint workSize = dim/NPEs;
  const uint workSizeRemainder = dim % NPEs;
  const uint myWorkSize = workSize + ((uint)myRank < workSizeRemainder ? 1 : 0);
  const size_t my_byte_dim = sizeof(double) * myWorkSize * dimWithEdges;
  const uint shift = myRank*workSize + ((uint)myRank < workSizeRemainder ? (uint)myRank : workSizeRemainder);
  double *matrix     = ( double* )malloc( my_byte_dim ); 
  double *matrix_new = ( double* )malloc( my_byte_dim );
  double *tmp_matrix;  

  double *firstRow = (double *)malloc(dimWithEdges * sizeof(double));
  double *lastRow = (double *)malloc(dimWithEdges * sizeof(double));
#if defined(SAVEGIF) || defined(SAVEPLOT) 
  double* firstRowSend = myRank ? NULL : firstRow;
  double* lastRowSend = myRank < NPEs-1 ? NULL : lastRow;
#endif
  MPI_Win firstRowWin, lastRowWin;
  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "same_size", "true");
  MPI_Info_set(info, "same_disp_unit", "true");
  MPI_Win_create(firstRow, dimWithEdges*sizeof(double), sizeof(double), info, MPI_COMM_WORLD, &firstRowWin);
  MPI_Win_create(lastRow, dimWithEdges*sizeof(double), sizeof(double), info, MPI_COMM_WORLD, &lastRowWin);
  t.initPar = end(&t);
  start(&t);
  init( matrix, matrix_new, myWorkSize, dimWithEdges, firstRow, lastRow, shift, myRank, NPEs);
  t.init = end(&t);
#ifdef DEBUG
  printMatrixThrSafe(matrix_new, myWorkSize, dimWithEdges, myRank, NPEs, firstRow, lastRowWin);
  printMatrixThrSafe(matrix, myWorkSize, dimWithEdges, myRank, NPEs, firstRow, lastRowWin);
#endif
#ifdef SAVEGIF
  save_gnuplot( matrix, myWorkSize, dimWithEdges, shift, 0, firstRowSend, lastRowSend);
#endif 

  // start algorithm
  MPI_Barrier(MPI_COMM_WORLD);
  t.evolveStart = MPI_Wtime();
  for(size_t it = 0; it < iterations; ++it )
  {
    start(&t);
    evolve( matrix, matrix_new, myWorkSize, dimWithEdges, firstRow, lastRow);
    t.update += end(&t);
    MPI_Barrier(MPI_COMM_WORLD);
    start(&t);
    if(myRank<NPEs-1){
      MPI_Win_lock(MPI_LOCK_EXCLUSIVE, next, MPI_MODE_NOCHECK, firstRowWin);
      MPI_Put(&matrix_new[(myWorkSize-1)*dimWithEdges], dimWithEdges, MPI_DOUBLE, next, 0, dimWithEdges, MPI_DOUBLE, firstRowWin);
      MPI_Win_unlock(next, firstRowWin);
    }
    if(myRank){
      MPI_Win_lock(MPI_LOCK_EXCLUSIVE, prev, MPI_MODE_NOCHECK, lastRowWin);
      MPI_Put(matrix_new, dimWithEdges, MPI_DOUBLE, prev, 0, dimWithEdges, MPI_DOUBLE, lastRowWin);
      MPI_Win_unlock(prev, lastRowWin);
    }
    t.comm += end(&t);
    tmp_matrix = matrix;
    matrix = matrix_new;
    matrix_new = tmp_matrix;
#ifdef DEBUG
    printMatrixThrSafe(matrix, myWorkSize, dimWithEdges, myRank, NPEs, firstRow, lastRowWin);
#endif
#ifdef SAVEGIF
    save_gnuplot( matrix, myWorkSize, dimWithEdges, shift, it+1, firstRowSend, lastRowSend);    
#endif
  }
  MPI_Barrier(MPI_COMM_WORLD);
  t.evolve = MPI_Wtime() - t.evolveStart;

#ifdef SAVEPLOT
  start(&t);
  save_gnuplot( matrix, myWorkSize, dimWithEdges, shift, 0, firstRowSend, lastRowSend);
  t.save = end(&t);
#endif
  t.total = MPI_Wtime() - t.programStart;
  printTimings(&t, myRank, NPEs);

  free( matrix );
  free( matrix_new );
  free( firstRow );
  free( lastRow );
  MPI_Win_free(&firstRowWin);
  MPI_Win_free(&lastRowWin);
  
  MPI_Finalize();
  return 0;
}
