CC=mpicc
INCLUDE=include
OBJDIR=obj
SRC=src
FUNC=base
PRINTDIR=--no-print-directory

FLAGS=-O3 -I$(INCLUDE)
OPT=-Wall -Wextra -march=native
OPENMP=-fopenmp
CBLAS=-DCBLAS -lopenblas
CUDA=-DCUDA -lcublas -lcudart -lmpi -L/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/lib

OBJECTS=$(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))

FLAGS+=-DDEBUG
#FLAGS+=-DPRINTTIME

base: FLAGS+=$(OPT) $(OPENMP)
base: main

blas: FLAGS+=$(CBLAS) $(OPT) $(OPENMP)
blas: main

cuda: FLAGS+=$(CUDA)
cuda: CC=nvcc
cuda: main

main: $(OBJECTS) main.c
	@$(CC) $(FLAGS) $^ -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	@mkdir -p $(OBJDIR)
	@$(CC) $(FLAGS) -c $^ -o $@

clean:
	@rm -rf main main.o
	@rm -rf $(OBJDIR)

git:
	@$(MAKE) clean $(PRINTDIR)
	@git add .
	@git commit -m "$(MS)"
	@git push

runBase: FUNC=base
runBase: run

runBlas: FUNC=blas
runBlas: run

runCuda: FUNC=cuda
runCuda: run
	
run:
	@$(MAKE) clean $(PRINTDIR)
	@$(MAKE) $(FUNC) $(PRINTDIR)
	@mpirun -np $(NP) ./main $(SZ)
