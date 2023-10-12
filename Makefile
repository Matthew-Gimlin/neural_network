CC = gcc
CFLAGS = -Wall -O2
SOURCES = main.c matrix.c activation.c initialization.c neural_net.c cost.c
HEADERS = matrix.h activation.h initialization.h neural_net.h cost.h
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
