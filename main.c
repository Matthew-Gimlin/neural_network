#include "matrix.h"
#include "neural_net.h"
#include "initialization.h"
#include "activation.h"
#include "cost.h"
#include <stdlib.h>
#include <stdio.h>

void printWeights(NeuralNet *net)
{
    for (size_t i = 0; i < net->layers - 1; ++i)
    {
        printf("Weight matrix %lu:\n", i);
        matPrint(&net->weights[i]);
    }
}

int main()
{
    size_t layerSizes[] = {8, 4, 4, 2};
    NeuralNet net;

    // Initialize the weights by sampling from a normal distribution.
    // Initialize the biases to zero.
    netInit(&net, 4, layerSizes, initNormalDist, NULL);

    printWeights(&net);

    Matrix *feats = (Matrix *)malloc(10 * sizeof(Matrix));
    for (size_t i = 0; i < 10; ++i)
    {
        matInit(&feats[i], 8, 1);
        feats[i].elements[i % 8] = 1.0f;
    }

    Matrix *labels = (Matrix *)malloc(10 * sizeof(Matrix));
    for (size_t i = 0; i < 10; ++i)
    {
        matInit(&labels[i], 2, 1);
        labels[i].elements[i % 2] = 1.0f;
    }

    netStochGradDesc(&net,
                     feats,
                     labels,
                     10,
                     actSigmoid,
                     actSigmoidDeriv,
                     costSquaredErrDeriv,
                     10,
                     3,
                     0.1);
    
    printWeights(&net);
    
    netFree(&net);
    for (size_t i = 0; i < 10; ++i)
    {
        matFree(&feats[i]);
        matFree(&labels[i]);
    }
    free(feats);
    free(labels);
    
    return 0;
}
