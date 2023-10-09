#include "neural_net.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

/**
 * @brief Creates a neural network.
 *
 * @param net An uninitialized neural network.
 * @param layers A number of layers, including the input and output layers.
 * @param layerSizes A number of neurons for each layer.
 * @param initWeights An initialization function for the weights.
 * @param initBiases An initialization function for the baises.
 */
void netInit(NeuralNet *net,
             size_t layers,
             size_t *layerSizes,
             InitFunc initWeights,
             InitFunc initBiases)
{
    net->layers = layers;
    net->layerSizes = (size_t *)malloc(layers * sizeof(size_t));
    for (size_t i = 0; i < layers; ++i)
    {
        net->layerSizes[i] = layerSizes[i];
    }

    net->weights = (Matrix *)malloc((layers - 1) * sizeof(Matrix));
    net->biases = (Matrix *)malloc((layers - 1) * sizeof(Matrix));
    for (size_t i = 0; i < layers - 1; ++i)
    {
        matInit(&net->weights[i], layerSizes[i + 1], layerSizes[i]);
        matInit(&net->biases[i], layerSizes[i + 1], 1);
        
        if (initWeights != NULL)
        {
            initWeights(&net->weights[i]);
        }
        if (initBiases != NULL)
        {
            initBiases(&net->biases[i]);
        }
    }
}

/**
 * @brief Frees the memory of a neural network.
 *
 * @param net An initialized neural network.
 */
void netFree(NeuralNet *net)
{
    free(net->layerSizes);
    for (size_t i = 0; i < net->layers - 1; ++i)
    {
        matFree(&net->weights[i]);
        matFree(&net->biases[i]);
    }
    free(net->weights);
    free(net->biases);
    
    net->layerSizes = NULL;
    net->weights = NULL;
    net->biases = NULL;
}

/**
 * @brief Predicts the label for a feature.
 *
 * @param net An initialized neural network.
 * @param features A feature matrix.
 * @param activation An activation function.
 * @return An output matrix.
 */
Matrix netPredict(NeuralNet *net, Matrix *features, ActivationFunc activation)
{
    Matrix prediction;
    matCopy(features, &prediction);
    for (size_t i = 0; i < net->layers - 1; ++i)
    {
        Matrix mul = matMul(&net->weights[i], &prediction);
        Matrix add = matAdd(&net->biases[i], &mul);
        matFree(&prediction);
        prediction = activation(&add);
        
        matFree(&mul);
        matFree(&add);
    }

    return prediction;
}
