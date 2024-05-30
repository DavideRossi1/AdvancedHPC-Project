#pragma once
#include <mpi.h>

struct Timer{
    double initTime;
    double updTime;
    double sendRecvTime;
    double evolveTime;
    double saveTime;
    double totalTime;

    double programStart;
    double evolveStart;
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

// double seconds();