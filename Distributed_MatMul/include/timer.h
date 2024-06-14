/**
 * @file timer.h
 * @author Davide Rossi
 * @brief header file for the timer functions, used to measure the time of the different phases of the program
 * @date 2024-06
 * 
 */
#pragma once
#include <mpi.h>

// Guard to avoid compiler to compile something as C++ code
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Timer struct, contains the results, in seconds, of the different phases of the program
 * 
 */
struct Timer{
    double initCuda;
    double init;
    double initComm;
    double gather;
    double resAlloc;
    double dgemm;
    double place;
    double mult;
    double total;
    
    double programStart;
    double multStart;
    double start;
};

/**
 * @brief Print the timings of the different phases of the program
 * 
 * @param t the timer struct where results are stored
 * @param myRank MPI rank of the executing process
 * @param NPEs Total number of MPI processes
 */
void printTimings(struct Timer* t, int myRank, int NPEs);

/**
 * @brief Start the timer
 * 
 * @param t the timer struct where results are stored
 */
inline void start(struct Timer* t){
    MPI_Barrier(MPI_COMM_WORLD);
    t->start = MPI_Wtime();
}

/**
 * @brief End the timer and return the time elapsed from the start
 * 
 * @param t the timer struct where results are stored
 * @return double the time elapsed from the start
 */
inline double end(struct Timer* t){
    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_Wtime()- t->start;
}

#ifdef __cplusplus
}
#endif
