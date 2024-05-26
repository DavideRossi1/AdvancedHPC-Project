#include <sys/time.h>
#include <stdio.h>
#include <mpi.h>

#include "timings.h"

void printTimings(struct Timings* t, int myRank, int NPEs){
    struct Timings maxT;
    struct Timings avgT;
    MPI_Reduce(&t->initTime,   &maxT.initTime,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->evolveTime, &maxT.evolveTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->saveTime,   &maxT.saveTime,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->totalTime,  &maxT.totalTime,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&t->initTime,   &avgT.initTime,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->evolveTime, &avgT.evolveTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->saveTime,   &avgT.saveTime,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->totalTime,  &avgT.totalTime,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(!myRank) {
        printf("%f;%f;%f;%f\n", maxT.initTime, maxT.evolveTime, maxT.saveTime, maxT.totalTime);
        printf("%f;%f;%f;%f\n", avgT.initTime/NPEs, avgT.evolveTime/NPEs, avgT.saveTime/NPEs, avgT.totalTime/NPEs);
    }
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
