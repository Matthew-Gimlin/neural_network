#include "activation.h"
#include "matrix.h"
#include <math.h>

/**
 * @brief Performs the sigmoid function.
 *
 * @param mat An initialized matrix.
 * @return A new result matrix.
 */
Matrix actSigmoid(Matrix *mat)
{
    Matrix result;
    matInit(&result, mat->rows, mat->columns);
    for (size_t i = 0; i < mat->rows * mat->columns; ++i)
    {
        result.elements[i] = 1.0f / (1.0f + expf(mat->elements[i]));
    }

    return result;
}

/**
 * @brief Performs the sigmoid function derivative.
 *
 * @param mat An initialized matrix.
 * @return A new result matrix.
 */
Matrix actSigmoidDeriv(Matrix *mat)
{
    Matrix result;
    matInit(&result, mat->rows, mat->columns);
    for (size_t i = 0; i < mat->rows * mat->columns; ++i)
    {
        float sigmoid = 1.0f / (1.0f + expf(mat->elements[i]));
        result.elements[i] = sigmoid * (1.0f - sigmoid);
    }

    return result;
}
