# inputs for the executables
CC=mpicc
INCLUDE = include
OBJDIR=obj
SRC=src
dim = 10
it  = 100

FLAGS=-O3 -I$(INCLUDE)
OPT=-Wall -Wextra -march=native
OPENMP=-fopenmp
CBLAS=-DCBLAS -lopenblas
CUDA=-DCUDA -lcublas -lcudart -lmpi -L/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/lib

OBJECTS=$(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))

#FLAGS+=-DDEBUG
FLAGS+=-DCONVERT
FLAGS+=-DPRINTTIME
#FLAGS+=-DPRINTMATRIX
FLAGS+=$(OPENMP) $(OPT)

jacobi.x: $(OBJECTS) jacobi.c
	@$(CC) $(FLAGS) $^ -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	@mkdir -p $(OBJDIR)
	@$(CC) $(FLAGS) -c $^ -o $@

clean:
	@rm -rf jacobi.x solution.*
	@rm -rf $(OBJDIR)

git:
	@$(MAKE) clean $(PRINTDIR)
	@git add .
	@git commit -m "$(MS)"
	@git push

plot:
	@gnuplot -p plot.plt

.PHONY: clean plot git

