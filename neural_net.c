#include "neural_net.h"
#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <stdio.h>

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
             NetInitFunc initWeights,
             NetInitFunc initBiases)
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
Matrix netPredict(NeuralNet *net,
                  Matrix *features, 
                  NetActivationFunc activation)
{
    Matrix prediction = matCopy(features);
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

/**
 * @brief Shuffles the order of the training data.
 *
 * @param trainingFeats A set of training features.
 * @param trainingLabels A set of labels for each training features.
 * @param trainingSize The number of training samples.
 */
void netShuffle(Matrix *trainingFeats,
                Matrix *trainingLabels,
                size_t trainingSize)
{
    srand(time(NULL));
    for (size_t i = trainingSize - 1; i > 0; --i)
    {
        size_t j = rand() % (i + 1);

        Matrix tempFeat = trainingFeats[i];
        trainingFeats[i] = trainingFeats[j];
        trainingFeats[j] = tempFeat;
        
        Matrix tempLabel = trainingLabels[i];
        trainingLabels[i] = trainingLabels[j];
        trainingLabels[j] = tempLabel;
    }
}

/**
 * @brief Performs stochastic gradient descent.
 *
 * @param net An initialized neural network.
 * @param trainingFeats A set of training features.
 * @param trainingLabels A set of labels for each training features.
 * @param trainingSize The number of training samples.
 * @param epochs A number of epochs.
 * @param miniBatchSize A number of training samples in a mini batch.
 * @param learningRate A learning rate.
 */
void netStochGradDesc(NeuralNet *net,
                      Matrix *trainingFeats,
                      Matrix *trainingLabels,
                      size_t trainingSize,
                      NetActivationFunc activation,
                      NetActivationFunc activationDeriv,
                      NetCostFunc costDeriv,
                      size_t epochs,
                      size_t miniBatchSize,
                      float learningRate)
{
    for (size_t i = 1; i <= epochs; ++i)
    {
        netShuffle(trainingFeats, trainingLabels, trainingSize);
        
        for (size_t j = 0; j < trainingSize; j += miniBatchSize)
        {
            if (j + miniBatchSize > trainingSize)
            {
                miniBatchSize = trainingSize - j;
            }

            netUpdateMiniBatch(net,
                               &trainingFeats[j],
                               &trainingLabels[j],
                               miniBatchSize,
                               activation,
                               activationDeriv,
                               costDeriv,
                               learningRate);
        }

        printf("Completed epoch %lu of %lu\n", i, epochs);
    }
}

void netUpdateMiniBatch(NeuralNet *net,
                        Matrix *miniBatchFeats,
                        Matrix *miniBatchLabels,
                        size_t miniBatchSize,
                        NetActivationFunc activation,
                        NetActivationFunc activationDeriv,
                        NetCostFunc costDeriv,
                        float learningRate)
{
    Matrix *weightGradientSums =
        (Matrix *)malloc((net->layers - 1) * sizeof(Matrix));
    Matrix *biasGradientSums =
        (Matrix *)malloc((net->layers - 1) * sizeof(Matrix));
    
    for (size_t i = 0; i < net->layers - 1; ++i)
    {
        matInit(&weightGradientSums[i],
                net->layerSizes[i + 1], 
                net->layerSizes[i]);
        matInit(&biasGradientSums[i], net->layerSizes[i + 1], 1);
    }

    for (size_t i = 0; i < miniBatchSize; ++i)
    {
        NetGradients gradients = netBackprop(net,
                                             &miniBatchFeats[i],
                                             &miniBatchLabels[i],
                                             activation,
                                             activationDeriv,
                                             costDeriv);

        for (size_t j = 0; j < net->layers - 1; ++j)
        {
            Matrix newWeightSum = matAdd(&weightGradientSums[j],
                                         &gradients.weightGrads[j]);
            Matrix newBiasSum = matAdd(&biasGradientSums[j],
                                       &gradients.biasGrads[j]);

            matFree(&weightGradientSums[j]);
            matFree(&biasGradientSums[j]);
            weightGradientSums[j] = newWeightSum;
            biasGradientSums[j] = newBiasSum;
            
            matFree(&gradients.weightGrads[j]);
            matFree(&gradients.biasGrads[j]);
        }
        
        free(gradients.weightGrads);
        free(gradients.biasGrads);
    }

    for (size_t i = 0; i < net->layers - 1; ++i)
    {
        Matrix weightGradAvgs = matScalarMul(&weightGradientSums[i],
                                             learningRate / miniBatchSize);
        Matrix biasGradAvgs = matScalarMul(&biasGradientSums[i],
                                           learningRate / miniBatchSize);
        Matrix newWeights = matSub(&net->weights[i], &weightGradAvgs);
        Matrix newBiases = matSub(&net->biases[i], &biasGradAvgs);

        matFree(&net->weights[i]);
        matFree(&net->biases[i]);
        net->weights[i] = newWeights;
        net->biases[i] = newBiases;

        matFree(&weightGradAvgs);
        matFree(&biasGradAvgs);
        matFree(&weightGradientSums[i]);
        matFree(&biasGradientSums[i]);
    }

    free(weightGradientSums);
    free(biasGradientSums);
}

/**
 * @brief Performs the backpropagation algorithm.
 *
 * @param net An initialized neural network.
 * @param features A feature matrix.
 * @param label The label for the feature matrix.
 * @param activation An activation function.
 * @param activationDeriv The derivative of the activation function.
 * @param costDeriv The derivative of a cost function.
 * @return Gradients of the weights and biases for each layer.
 */
NetGradients netBackprop(NeuralNet *net,
                         Matrix *features,
                         Matrix *label,
                         NetActivationFunc activation,
                         NetActivationFunc activationDeriv,
                         NetCostFunc costDeriv)
{
    Matrix *weightGradients =
        (Matrix *)malloc((net->layers - 1) * sizeof(Matrix));
    Matrix *biasGradients =
        (Matrix *)malloc((net->layers - 1) * sizeof(Matrix));
    
    Matrix act = matCopy(features);
    Matrix *activationOutputs = (Matrix *)malloc(net->layers * sizeof(Matrix));
    activationOutputs[0] = act;
    Matrix *activationInputs =
        (Matrix *)malloc((net->layers - 1) * sizeof(Matrix));

    // Forward pass.
    for (size_t i = 0; i < net->layers - 1; ++i)
    {
        Matrix mul = matMul(&net->weights[i], &act);
        Matrix add = matAdd(&mul, &net->biases[i]);
        
        activationInputs[i] = add;
        act = activation(&add);
        activationOutputs[i + 1] = act;

        matFree(&mul);
    }
    
    Matrix costDerivOutput = costDeriv(&activationOutputs[net->layers - 1],
                                       label);
    Matrix actDeriv = activationDeriv(&activationInputs[net->layers - 2]);
    Matrix delta = matElementMul(&costDerivOutput, &actDeriv);
    biasGradients[net->layers - 2] = delta;
    Matrix transpose = matTranspose(&activationOutputs[net->layers - 2]);
    weightGradients[net->layers - 2] = matMul(&delta, &transpose);

    matFree(&costDerivOutput);
    matFree(&actDeriv);
    matFree(&transpose);

    // Backward pass.
    for (size_t i = net->layers - 3; i < net->layers; --i)
    {
        actDeriv = activationDeriv(&activationInputs[i]);
        Matrix weightTranspose = matTranspose(&net->weights[i + 1]);
        Matrix mul = matMul(&weightTranspose, &delta);
        delta = matElementMul(&mul, &actDeriv);

        biasGradients[i] = delta;
        transpose = matTranspose(&activationOutputs[i]);
        weightGradients[i] = matMul(&delta, &transpose);

        matFree(&actDeriv);
        matFree(&weightTranspose);
        matFree(&mul);
        matFree(&transpose);
    }

    for (size_t i = 0; i < net->layers - 1; ++i)
    {
        matFree(&activationOutputs[i]);
        matFree(&activationInputs[i]);
    }
    matFree(&activationOutputs[net->layers - 1]);
    free(activationOutputs);
    free(activationInputs);
    
    return (NetGradients){weightGradients, biasGradients};
}
