# Neural Network

This project implements a feed-forward neural network in C.

## Demonstration

The `main.c` file contains sample code that trains a neural network to recognize
handwritten digits. It uses the MNIST dataset stored in the `data/` folder, and 
built using the `Makefile` file.

```
$ make
gcc -Wall -O2 -c main.c -o main.o -lm
gcc -Wall -O2 -c matrix.c -o matrix.o -lm
gcc -Wall -O2 -c activation.c -o activation.o -lm
gcc -Wall -O2 -c initialization.c -o initialization.o -lm
gcc -Wall -O2 -c neural_net.c -o neural_net.o -lm
gcc -Wall -O2 -c cost.c -o cost.o -lm
gcc main.o matrix.o activation.o initialization.o neural_net.o cost.o -o net -lm

$ ./net
9415 correct of 10000
Accuracy: 0.94
```
