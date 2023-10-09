#include "matrix.h"
#include "neural_net.h"
#include "initialization.h"
#include "activation.h"
#include <stdlib.h>

int main()
{
    size_t layerSizes[] = {8, 4, 4, 2};
    NeuralNet net;

    // Initialize the weights by sampling from a normal distribution.
    // Initialize the biases to zero.
    netInit(&net, 4, layerSizes, initNormalDist, NULL);

    Matrix feature;
    matInit(&feature, 8, 1);
    feature.elements[0] = 1.0;

    // Make a random prediction.
    Matrix prediction = netPredict(&net, &feature, actSigmoid);
    matPrint(&prediction);

    netFree(&net);
    matFree(&feature);
    matFree(&prediction);
    
    return 0;
}
