# Neural Network

This project implements a feed-forward neural network in C.

## Demonstration

The `main.c` file contains sample code that trains a neural network to recognize
handwritten digits. It uses the MNIST dataset stored in the `data/` folder, and 
is built using the `Makefile` file.

```
$ make
gcc -Wall -O2 -c main.c -o main.o -lm
gcc -Wall -O2 -c src/matrix.c -o src/matrix.o -lm
gcc -Wall -O2 -c src/activation.c -o src/activation.o -lm
gcc -Wall -O2 -c src/initialization.c -o src/initialization.o -lm
gcc -Wall -O2 -c src/neural_net.c -o src/neural_net.o -lm
gcc -Wall -O2 -c src/cost.c -o src/cost.o -lm
gcc main.o src/matrix.o src/activation.o src/initialization.o src/neural_net.o src/cost.o -o net -lm

$ ./net
Training...
Testing...
9445 correct of 10000
Accuracy: 0.94
```
