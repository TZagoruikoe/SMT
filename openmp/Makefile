#Compiler
CC= gcc

#Flags
CFLAGS= -Wall -Wextra -g -O2 -fopenmp
LDLIBS= -lm

#Sources
PROG= smt.c
TARGET= smt 

all: $(TARGET)
$(TARGET): $(PROG)
clean:
	rm -f $(TARGET) *.o *.out