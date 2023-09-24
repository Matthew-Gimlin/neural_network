#ifndef MATRIX_H
#define MATRIX_H

#include "stddef.h"

typedef struct
{
    size_t rows, columns;
    float *elements;
}
Matrix;

void matInit(Matrix *mat, size_t rows, size_t columns);
void matCopy(Matrix *matSrc, Matrix *matDest);
void matSet(Matrix *mat, float value);
void matFree(Matrix *mat);

void matPrint(Matrix *mat);

Matrix matTranspose(Matrix *mat);
Matrix matAdd(Matrix *matA, Matrix *matB);
Matrix matSub(Matrix *matA, Matrix *matB);
Matrix matMul(Matrix *matA, Matrix *matB);
Matrix matElemMul(Matrix *matA, Matrix *matB);

#endif
