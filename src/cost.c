#include "cost.h"
#include "matrix.h"

Matrix costSquaredErr(Matrix *prediction, Matrix *label)
{
    return (Matrix){0, 0, NULL};
}

Matrix costSquaredErrDeriv(Matrix *prediction, Matrix *label)
{
    return matSub(prediction, label);
}
