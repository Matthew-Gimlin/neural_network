CC = gcc
CFLAGS = -Wall -O2
SOURCES = main.c src/matrix.c src/activation.c src/initialization.c src/neural_net.c src/cost.c
HEADERS = src/matrix.h src/activation.h src/initialization.h src/neural_net.h src/cost.h
OBJECTS = $(SOURCES:.c=.o)
LIBRARIES = -lm
EXECUTABLE = net

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(EXECUTABLE) $(LIBRARIES)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@ $(LIBRARIES)

clean:
	rm -f $(EXECUTABLE) $(OBJECTS)
