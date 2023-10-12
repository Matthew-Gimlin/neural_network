#include "matrix.h"
#include "neural_net.h"
#include "initialization.h"
#include "activation.h"
#include "cost.h"
#include <stdlib.h>
#include <stdio.h>

int main()
{
    size_t layerSizes[] = {8, 4, 4, 2};
    NeuralNet net;

    // Initialize the weights by sampling from a normal distribution.
    // Initialize the biases to zero.
    netInit(&net, 4, layerSizes, initNormalDist, NULL);

    Matrix feats;
    matInit(&feats, 8, 1);
    feats.elements[0] = 1.0f;

    // Make a random prediction.
    Matrix pred = netPredict(&net, &feats, actSigmoid);
    printf("Prediction:\n");
    matPrint(&pred);

    Matrix label;
    matInit(&label, 2, 1);
    label.elements[0] = 1.0f;

    NetGradients grads = netBackprop(&net,
                                     &feats,
                                     &label,
                                     actSigmoid,
                                     actSigmoidDeriv,
                                     costSquaredErrDeriv);

    for (size_t i = 0; i < 3; ++i)
    {
        printf("Gradient %lu:\n", i);
        matPrint(&grads.weights[i]);

        matFree(&grads.weights[i]);
        matFree(&grads.biases[i]);
    }
    free(grads.weights);
    free(grads.biases);
    
    netFree(&net);
    matFree(&feats);
    matFree(&pred);
    
    return 0;
}
