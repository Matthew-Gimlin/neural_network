#include "matrix.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

/**
 * @brief Initializes a Matrix struct. Allocates a flattened 2D array filled
 *        with the value 0.0.
 *
 * @param mat The matrix to initialize.
 * @param rows The number of rows for the matrix.
 * @param columns The number of columns for the matrix.
 */
void matInit(Matrix *mat, size_t rows, size_t columns)
{
    mat->rows = rows;
    mat->columns = columns;
    mat->elements = (float *)calloc(rows * columns, sizeof(float));
}

/**
 * @brief Copies the contents of a source matrix into a destination matrix.
 *
 * @param src The source matrix.
 * @param dest The destination matrix.
 */
void matCopy(Matrix *src, Matrix *dest)
{
    matInit(dest, src->rows, src->columns);
    for (size_t i = 0; i < src->rows * src->columns; ++i)
    {
        dest->elements[i] = src->elements[i];
    }
}

/**
 * @brief Frees the internal array of a matrix.
 *
 * @param mat The matrix to free.
 */
void matFree(Matrix *mat)
{
    if (mat == NULL)
    {
        return;
    }
    
    free(mat->elements);
}

/**
 * @brief Sets all elements of a matrix to a value.
 *
 * @param mat The matrix to set.
 * @param value The value of the elements.
 */
void matSet(Matrix *mat, float value)
{
    for (size_t i = 0; i < mat->rows * mat->columns; ++i)
    {
        mat->elements[i] = value;
    }
}

/**
 * @brief Prints the elements of a matrix.
 *
 * @param mat The matix to print.
 */
void matPrint(Matrix *mat)
{
    for (size_t i = 0; i < mat->rows; ++i)
    {
        printf("[ ");
        for (size_t j = 0; j < mat->columns; ++j)
        {
            printf("%.2f", mat->elements[i * mat->columns + j]);
        }
        printf(" ]\n");
    }
}

/**
 * @brief Tranposes a matrix.
 *
 * @param mat The matrix to transpose.
 * @return A new matrix that stores the result.
 */
Matrix matTranspose(Matrix *mat)
{
    Matrix result;
    matInit(&result, mat->columns, mat->rows);

    for (size_t i = 0; i < result.rows; ++i)
    {
        for (size_t j = 0; j < result.columns; ++j)
        {
            result.elements[i * result.columns + j] =
                mat->elements[j * mat->columns + i];
        }
    }

    return result;
}

/**
 * @brief Performs element-wise matrix addition. The two matrices must have the 
 *        same dimensions.
 *
 * @param matA The first matrix.
 * @param matB The second matrix.
 * @return A new matrix that stores the result.
 */
Matrix matAdd(Matrix *matA, Matrix *matB)
{
    if (matA->rows != matB->rows || matA->columns != matB->columns)
    {
        fprintf(stderr,
                "Error: Cannot add matrices (%lu, %lu) and (%lu, %lu)\n",
                matA->rows, matA->columns,
                matB->rows, matB->columns);

        return (Matrix){0, 0, NULL};
    }
    
    Matrix result;
    matInit(&result, matA->rows, matA->columns);
    for (size_t i = 0; i < result.rows * result.columns; ++i)
    {
        result.elements[i] = matA->elements[i] + matB->elements[i];
    }
    
    return result;
}

/**
 * @brief Performs element-wise matrix subtraction. The two matrices must have
 *        the same dimensions.
 *
 * @param matA The first matrix.
 * @param matB The second matrix.
 * @return A new matrix that stores the result.
 */
Matrix matSub(Matrix *matA, Matrix *matB)
{
    if (matA->rows != matB->rows || matA->columns != matB->columns)
    {
        fprintf(stderr,
                "Error: Cannot subtract matrices (%lu, %lu) and (%lu, %lu)\n",
                matA->rows, matA->columns,
                matB->rows, matB->columns);
        
        return (Matrix){0, 0, NULL};
    }
    
    Matrix result;
    matInit(&result, matA->rows, matA->columns);
    for (size_t i = 0; i < result.rows * result.columns; ++i)
    {
        result.elements[i] = matA->elements[i] + matB->elements[i];
    }
    
    return result;
}

/**
 * @brief Performs matrix multiplication. The columns of the first matrix must
 *        match the rows of the second.
 *
 * @param matA The first matrix.
 * @param matB The second matrix.
 * @return A new matrix that stores the result.
 */
Matrix matMul(Matrix *matA, Matrix *matB)
{
    if (matA->columns != matB->rows)
    {
        fprintf(stderr,
                "Error: Cannot multiply matrices (%lu, %lu) and (%lu, %lu)\n",
                matA->rows, matA->columns,
                matB->rows, matB->columns);
        
        return (Matrix){0, 0, NULL};
    }
    
    Matrix result;
    matInit(&result, matA->rows, matB->columns);
    for (size_t i = 0; i < result.rows; ++i)
    {
        for (size_t j = 0; j < result.columns; ++j)
        {
            for (size_t k = 0; k < matA->columns; ++k)
            {
                result.elements[i * result.columns + j] +=
                    matA->elements[i * matA->columns + k] *
                    matB->elements[k * matB->columns + j];
            }
        }
    }

    return result;
}

/**
 * @brief Performs element-wise matrix multiplication. The two matrices must 
 *        have the same dimensions.
 *
 * @param matA The first matrix.
 * @param matB The second matrix.
 * @return A new matrix that stores the result.
 */
Matrix matElemMul(Matrix *matA, Matrix *matB)
{
    if (matA->rows != matB->rows || matA->columns != matB->columns)
    {
        fprintf(stderr,
                "Error: Cannot multiply matrices (%lu, %lu) and (%lu, %lu)\n",
                matA->rows, matA->columns,
                matB->rows, matB->columns);

        return (Matrix){0, 0, NULL};
    }
    
    Matrix result;
    matInit(&result, matA->rows, matA->columns);
    for (size_t i = 0; i < result.rows * result.columns; ++i)
    {
        result.elements[i] = matA->elements[i] * matB->elements[i];
    }
    
    return result;
}
