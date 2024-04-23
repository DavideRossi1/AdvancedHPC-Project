
void readBlockFromMatrix(double *block, double *matrix, int nRows, int nCols, int N, int startingCol);

void placeBlockInMatrix(double *block, double *matrix, int nRows, int nCols, int N, int startingCol);

void matMul(double *A, double *B, double *C, int nRowsA, int nColsARowsB, int nColsB, int startingCol);

void matMulCblas(double *A, double *B, double *C, int nRowsA, int nColsARowsB, int nColsB, int startingCol);
