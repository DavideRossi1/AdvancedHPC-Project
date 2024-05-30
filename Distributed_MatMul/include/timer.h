#pragma once
#include <mpi.h>

struct Timer{
    double initTime;
    double initCommTime;
    double gatherTime;
    double resAllocTime;
    double dgemmTime;
    double placeTime;
    double multTime;
    double totalTime;
    
    double programStart;
    double multStart;
    double start;
};

void printTimings(struct Timer* t, int myRank, int NPEs);

inline void start(struct Timer* t){
    MPI_Barrier(MPI_COMM_WORLD);
    t->start = MPI_Wtime();
}

inline double end(struct Timer* t){
    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_Wtime()- t->start;
}
