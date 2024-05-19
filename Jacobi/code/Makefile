# inputs for the executables

dim = 10
it  = 100
INCLUDE = include
OBJDIR=obj
SRC=src
CC=mpicc
CFLAGS=-O3 -I$(INCLUDE)
OBJECTS=$(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))


SRCS=$(wildcard *.c)
EXE=$(SRCS:.c=.x)

all: $(EXE)

run: clean all
	./$(EXE) $(dim) $(it)

jacobi.x: $(OBJECTS) jacobi.c
	@$(CC) $(FLAGS) $^ -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	@mkdir -p $(OBJDIR)
	@$(CC) $(FLAGS) -c $^ -o $@

clean:
	@rm -f *~ $(EXE) solution.dat 

plot:
	@gnuplot -p plot.plt

.PHONY: clean plot all

