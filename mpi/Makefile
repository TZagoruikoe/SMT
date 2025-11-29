#Compiler
CC= mpicc

#Flags
CFLAGS= -Wall -Wextra -g -O2
LDLIBS= -lm

#Sources
PROG= smt.c
TARGET= smt 

all: $(TARGET)
$(TARGET): $(PROG)
clean:
	rm -f $(TARGET) *.o *.out