#include "src/matrix.h"
#include "src/neural_net.h"
#include "src/initialization.h"
#include "src/activation.h"
#include "src/cost.h"
#include <stdlib.h>
#include <stdio.h>

Matrix *loadFeatures(const char *fileName, size_t samples)
{
    FILE *file = fopen(fileName, "rb");

    // Skip the file header.
    fseek(file, 16, SEEK_SET);

    Matrix *data = (Matrix *)malloc(samples * sizeof(Matrix));
    for (size_t i = 0; i < samples; ++i)
    {
        matInit(&data[i], 28*28, 1);
        for (size_t j = 0; j < 28*28; ++j)
        {
            data[i].elements[j] = (unsigned char)fgetc(file) / 255.0f;
        }
    }

    return data;
}

Matrix *loadLabels(const char *fileName, size_t samples)
{
    FILE *file = fopen(fileName, "rb");

    // Skip the file header.
    fseek(file, 8, SEEK_SET);

    Matrix *data = (Matrix *)malloc(samples * sizeof(Matrix));
    for (size_t i = 0; i < samples; ++i)
    {
        matInit(&data[i], 10, 1);
        data[i].elements[fgetc(file)] = 1.0;
    }

    return data;
}

int main()
{
    // Set up the neural network.
    const size_t layers = 4;
    size_t layerSizes[] = {28*28, 16, 16, 10};
    NeuralNet net;
    netInit(&net, layers, layerSizes, initNormalDist, NULL);

    // Train the neural network on the MNIST dataset.
    const size_t trainingSize = 60000;
    Matrix *trainingFeats = loadFeatures("./data/train-images-idx3-ubyte", trainingSize);
    Matrix *trainingLabels = loadLabels("./data/train-labels-idx1-ubyte", trainingSize);
    netTrain(&net,
             trainingFeats,
             trainingLabels,
             trainingSize,
             actSigmoid,
             actSigmoidDeriv,
             costSquaredErrDeriv,
             30,
             10,
             2.0f);

    // Test the neural network.
    const size_t testingSize = 10000;
    Matrix *testingFeats = loadFeatures("./data/t10k-images-idx3-ubyte", testingSize);
    Matrix *testingLabels = loadLabels("./data/t10k-labels-idx1-ubyte", testingSize);
    size_t correct = netTest(&net,
                             testingFeats,
                             testingLabels,
                             testingSize,
                             actSigmoid);

    // Output the test results.
    float accuracy = (float)correct / testingSize;
    printf("%lu correct of %lu\n", correct, testingSize);
    printf("Accuracy: %.2f\n", accuracy);
    
    // Free all allocated memory.
    for (size_t i = 0; i < trainingSize; ++i)
    {
        matFree(&trainingFeats[i]);
        matFree(&trainingLabels[i]);
    }
    for (size_t i = 0; i < testingSize; ++i)
    {
        matFree(&testingFeats[i]);
        matFree(&testingLabels[i]);
    }
    free(trainingFeats);
    free(trainingLabels);
    free(testingFeats);
    free(testingLabels);
    netFree(&net);
    
    return 0;
}
