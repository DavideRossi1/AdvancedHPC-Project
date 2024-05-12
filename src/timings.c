#include <stdio.h>
#include <mpi.h>
#include <timings.h>

void printTimings(struct Timings* t, int myRank){
    struct Timings maxT;
    MPI_Reduce(&t->initTime, &maxT.initTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->initCommTime, &maxT.initCommTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->gatherTime, &maxT.gatherTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->resAllocTime, &maxT.resAllocTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->dgemmTime, &maxT.dgemmTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->placeTime, &maxT.placeTime,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->multTime, &maxT.multTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t->totalTime, &maxT.totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(!myRank) printf("%f;%f;%f;%f;%f;%f;%f;%f\n", maxT.initTime,maxT.initCommTime,maxT.gatherTime,maxT.resAllocTime,maxT.dgemmTime,maxT.placeTime,maxT.multTime,maxT.totalTime);
}