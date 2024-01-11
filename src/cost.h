#ifndef COST_H
#define COST_H

#include "matrix.h"

Matrix costSquaredErr(Matrix *prediction, Matrix *label);
Matrix costSquaredErrDeriv(Matrix *prediction, Matrix *label);

#endif
