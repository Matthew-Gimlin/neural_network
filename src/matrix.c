#include "matrix.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

/**
 * @brief Creates a zero matrix.
 *
 * @param mat An uninitialized matrix.
 * @param rows A number of rows.
 * @param columns A number of columns.
 */
void matInit(Matrix *mat, size_t rows, size_t columns)
{
    mat->rows = rows;
    mat->columns = columns;
    mat->elements = (float *)calloc(rows * columns, sizeof(float));
}

/**
 * @brief Performs a deep copy.
 *
 * @param mat An initialized matrix.
 * @return A new matrix copy.
 */
Matrix matCopy(Matrix *mat)
{
    Matrix copy;
    matInit(&copy, mat->rows, mat->columns);
    for (size_t i = 0; i < mat->rows * mat->columns; ++i)
    {
        copy.elements[i] = mat->elements[i];
    }
    
    return copy;
}

/**
 * @brief Frees the elements of a matrix.
 *
 * @param mat An initialized matrix.
 */
void matFree(Matrix *mat)
{
    free(mat->elements);
    mat->elements = NULL;
}

/**
 * @brief Fills a matrix with a value.
 *
 * @param mat An initialized matrix.
 * @param value A value for the matrix elements.
 */
void matSet(Matrix *mat, float value)
{
    for (size_t i = 0; i < mat->rows * mat->columns; ++i)
    {
        mat->elements[i] = value;
    }
}

/**
 * @brief Prints a matrix.
 *
 * @param mat An initialized matrix.
 */
void matPrint(Matrix *mat)
{
    for (size_t i = 0; i < mat->rows; ++i)
    {
        printf("[ ");
        for (size_t j = 0; j < mat->columns; ++j)
        {
            printf("%f ", mat->elements[i * mat->columns + j]);
        }
        printf("]\n");
    }
}

/**
 * @brief Finds the maximum element.
 *
 * @param mat An initialized matrix.
 * @return The index of the maximum element.
 */
size_t matMaxElement(Matrix *mat)
{
    size_t maxElemIndex = 0;
    for (size_t i = 0; i < mat->rows * mat->columns; ++i)
    {
        if (mat->elements[i] >= mat->elements[maxElemIndex])
        {
            maxElemIndex = i;
        }
    }

    return maxElemIndex;
}

/**
 * @brief Tranposes a matrix.
 *
 * @param mat An initialized matrix.
 * @return A new result matrix.
 */
Matrix matTranspose(Matrix *mat)
{
    Matrix result;
    matInit(&result, mat->columns, mat->rows);

    for (size_t i = 0; i < result.rows; ++i)
    {
        for (size_t j = 0; j < result.columns; ++j)
        {
            result.elements[i * result.columns + j] = mat->elements[j * mat->columns + i];
        }
    }

    return result;
}

/**
 * @brief Performs matrix addition.
 *
 * @param a An initialized matrix.
 * @param b An initialized matrix.
 * @return A new result matrix.
 */
Matrix matAdd(Matrix *a, Matrix *b)
{
    // Must have the same number of elements to add.
    if (a->rows != b->rows || a->columns != b->columns)
    {
        fprintf(stderr,
                "Error: Cannot add matrices (%lu, %lu) and (%lu, %lu)\n",
                a->rows, a->columns,
                b->rows, b->columns);

        return (Matrix){0, 0, NULL};
    }
    
    Matrix result;
    matInit(&result, a->rows, a->columns);
    for (size_t i = 0; i < result.rows * result.columns; ++i)
    {
        result.elements[i] = a->elements[i] + b->elements[i];
    }
    
    return result;
}

/**
 * @brief Performs matrix subtraction (a minus b).
 *
 * @param a An initialized matrix.
 * @param b An initialized matrix.
 * @return A new result matrix.
 */
Matrix matSub(Matrix *a, Matrix *b)
{
    // Must have the same number of elements to subtract.
    if (a->rows != b->rows || a->columns != b->columns)
    {
        fprintf(stderr,
                "Error: Cannot subtract matrices (%lu, %lu) and (%lu, %lu)\n",
                a->rows, a->columns,
                b->rows, b->columns);
        
        return (Matrix){0, 0, NULL};
    }
    
    Matrix result;
    matInit(&result, a->rows, a->columns);
    for (size_t i = 0; i < result.rows * result.columns; ++i)
    {
        result.elements[i] = a->elements[i] - b->elements[i];
    }
    
    return result;
}

/**
 * @brief Performs matrix multiplication (a times b).
 *
 * @param a An initialized matrix.
 * @param b An initialized matrix.
 * @return A new result matrix.
 */
Matrix matMul(Matrix *a, Matrix *b)
{
    // Required for matrix multiplication.
    if (a->columns != b->rows)
    {
        fprintf(stderr,
                "Error: Cannot multiply matrices (%lu, %lu) and (%lu, %lu)\n",
                a->rows, a->columns,
                b->rows, b->columns);
        
        return (Matrix){0, 0, NULL};
    }
    
    Matrix result;
    matInit(&result, a->rows, b->columns);
    for (size_t i = 0; i < result.rows; ++i)
    {
        for (size_t j = 0; j < result.columns; ++j)
        {
            for (size_t k = 0; k < a->columns; ++k)
            {
                result.elements[i * result.columns + j] +=
                    a->elements[i * a->columns + k] * b->elements[k * b->columns + j];
            }
        }
    }

    return result;
}

/**
 * @brief Performs element-wise matrix multiplication.
 *
 * @param a An initialized matrix.
 * @param b An initialized matrix.
 * @return A new result matrix.
 */
Matrix matElementMul(Matrix *a, Matrix *b)
{
    // Must have the same number of elements to multiply.
    if (a->rows != b->rows || a->columns != b->columns)
    {
        fprintf(stderr,
                "Error: Cannot multiply matrices (%lu, %lu) and (%lu, %lu)\n",
                a->rows, a->columns,
                b->rows, b->columns);

        return (Matrix){0, 0, NULL};
    }
    
    Matrix result;
    matInit(&result, a->rows, a->columns);
    for (size_t i = 0; i < result.rows * result.columns; ++i)
    {
        result.elements[i] = a->elements[i] * b->elements[i];
    }
    
    return result;
}

/**
 * @brief Performs scalar multiplication.
 *
 * @param mat An initialized matrix.
 * @param scalar A scalar.
 * @return A new result matrix.
 */
Matrix matScalarMul(Matrix *mat, float scalar)
{
    Matrix result;
    matInit(&result, mat->rows, mat->columns);
    for (size_t i = 0; i < result.rows * result.columns; ++i)
    {
        result.elements[i] = mat->elements[i] * scalar;
    }
    
    return result;
}
