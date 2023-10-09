#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stdlib.h>
#include "matrix.h"

typedef void (*InitFunc)(Matrix *);
typedef Matrix (*ActivationFunc)(Matrix *);

typedef struct
{
    size_t layers;
    size_t *layerSizes;
    Matrix *weights, *biases;
}
NeuralNet;

void netInit(NeuralNet *net,
             size_t layers,
             size_t *layerSizes,
             InitFunc initWeights,
             InitFunc initBiases);
void netFree(NeuralNet *net);

Matrix netPredict(NeuralNet *net, Matrix *features, ActivationFunc activation);

#endif
