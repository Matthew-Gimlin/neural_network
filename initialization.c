#include "matrix.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

/**
 * @brief Fill a matrix by sampling from a normal distribution.
 *
 * @param mat An initialized matrix.
 */
void initNormalDist(Matrix *mat)
{
    srand(time(NULL));
    for (size_t i = 0; i < mat->rows * mat->columns; ++i)
    {
        // Use the Box-Muller transform to sample from a normal distribution.
        float uniform1 = (float)rand() / RAND_MAX;
        float uniform2 = (float)rand() / RAND_MAX;
        float normal = 
            sqrtf(-2.0f * logf(uniform1)) * cosf(2.0f * M_PI * uniform2);

        mat->elements[i] = normal;
    }
}
