#include <string.h>

#include "../include/initUtilities.h"

void init(double* matrix, int myPID, int N, int workSize, int NPEs)
{
    int shift = 0;
    for (int i = 0; i < myPID; i++){
        shift += N*(N/NPEs + (i < N % NPEs ? 1 : 0));
    }
    for (int i = 0; i < workSize*N; i++){
        matrix[i] = i + shift;
    }
}

void initID(double* matrix, int myPID, int N, int workSize, int NPEs)
{
    memset(matrix, 0, workSize*N*sizeof(double));
    int shift = myPID*workSize + (myPID < N%NPEs ? 0 : N%NPEs);
    for (int i = 0; i < workSize; i++) 
        matrix[i*N + shift + i] = 1;
}

