LFLAGS=-lm
CC=mpicc

SRCS=$(wildcard *.c)
PROGS = $(patsubst %.c,%,$(SRCS))

all: $(PROGS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LFLAGS)

clean: 
	rm -f $(PROGS)

.PHONY: all