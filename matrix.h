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
void matCopy(Matrix *src, Matrix *dest);
void matSet(Matrix *mat, float value);
void matFree(Matrix *mat);

void matPrint(Matrix *mat);

Matrix matTranspose(Matrix *mat);
Matrix matAdd(Matrix *a, Matrix *b);
Matrix matSub(Matrix *a, Matrix *b);
Matrix matMul(Matrix *a, Matrix *b);
Matrix matElemMul(Matrix *a, Matrix *b);

#endif
