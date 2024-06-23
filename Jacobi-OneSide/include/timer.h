/**
 * @file timer.h
 * @author Davide Rossi
 * @brief Header file for the timer functions, used to measure the time of the different phases of the program
 * @date 2024-06
 * 
 */
#pragma once
#include <mpi.h>

/**
 * @brief Timer struct, contains the results, in seconds, of the different phases of the program
 * 
 */
struct Timer{
    double initPar;
    double init;
    double update;
    double comm;
    double evolve;
    double save;
    double total;

    double programStart;
    double evolveStart;
    double start;
};


/**
 * @brief Print the timings of the different phases of the program
 * 
 * @param t the Timer struct containing the timings
 * @param myRank MPI rank of the executing process
 * @param NPEs total number of MPI processes
 */
void printTimings(struct Timer* t, int myRank, int NPEs);


/**
 * @brief Start the timer
 * 
 * @param t the Timer struct to start
 */
inline void start(struct Timer* t){
    MPI_Barrier(MPI_COMM_WORLD);
    t->start = MPI_Wtime();
}


/**
 * @brief End the timer and return the time elapsed from the start
 * 
 * @param t the Timer struct to end
 * @return double: the time elapsed from the start
 */
inline double end(struct Timer* t){
    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_Wtime()- t->start;
}
