#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stdlib.h>
#include "matrix.h"

typedef void (*NetInitFunc)(Matrix *);
typedef Matrix (*NetActivationFunc)(Matrix *);
typedef Matrix (*NetCostFunc)(Matrix *, Matrix *);

typedef struct
{
    size_t layers;
    size_t *layerSizes;
    Matrix *weights, *biases;
}
NeuralNet;

typedef struct
{
    Matrix *weightGrads, *biasGrads;
}
NetGradients;

void netShuffle(Matrix *trainingFeats,
                Matrix *trainingLabels,
                size_t trainingSize);

void netInit(NeuralNet *net,
             size_t layers,
             size_t *layerSizes,
             NetInitFunc initWeights,
             NetInitFunc initBiases);
void netFree(NeuralNet *net);

Matrix netPredict(NeuralNet *net,
                  Matrix *features,
                  NetActivationFunc activation);
void netStochGradDesc(NeuralNet *net,
                      Matrix *trainingFeats,
                      Matrix *trainingLabels,
                      size_t trainingSize,
                      NetActivationFunc activation,
                      NetActivationFunc activationDeriv,
                      NetCostFunc costDeriv,
                      size_t epochs,
                      size_t miniBatchSize,
                      float learningRate);
void netUpdateMiniBatch(NeuralNet *net,
                        Matrix *miniBatchFeats,
                        Matrix *miniBatchLabels,
                        size_t miniBatchSize,
                        NetActivationFunc activation,
                        NetActivationFunc activationDeriv,
                        NetCostFunc costDeriv,
                        float learningRate);
NetGradients netBackprop(NeuralNet *net,
                         Matrix *features,
                         Matrix *label,
                         NetActivationFunc activation,
                         NetActivationFunc activationDeriv,
                         NetCostFunc costDeriv);

#endif
