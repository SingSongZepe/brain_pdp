LFLAGS=-lm
CC=mpif90

SRCS=$(wildcard *.F90)
PROGS = $(patsubst %.F90,%,$(SRCS))

all: $(PROGS)

%: %.F90
	$(CC) $(CFLAGS) -o $@ $< $(LFLAGS)

clean: 
	rm -f $(PROGS)

.PHONY: all