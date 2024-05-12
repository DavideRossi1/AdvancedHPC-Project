CC=mpicc
INCLUDE=include
OBJDIR=obj
SRC=src

FLAGS=-O3 -I$(INCLUDE)
OPT=-Wall -Wextra -march=native
OPENMP=-fopenmp
CBLAS=-DCBLAS -lopenblas
CUDA=-DCUDA -lcublas -lcudart -lmpi -L/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/lib

OBJECTS=$(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))

FLAGS+=-DDEBUG
#FLAGS+=-DPRINTTIME
FLAGS+=-DPRINTMATRIX

base: FLAGS+=$(OPT) $(OPENMP)
base: main
	@$(MAKE) main --no-print-directory

blas: FLAGS+=$(CBLAS) $(OPT) $(OPENMP)
blas: main
	@$(MAKE) main --no-print-directory

cuda: FLAGS+=$(CUDA)
cuda: CC=nvcc
cuda: main
	@$(MAKE) main --no-print-directory

main: $(OBJECTS) main.c
	@$(CC) $(FLAGS) $^ -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	@mkdir -p $(OBJDIR)
	@$(CC) $(FLAGS) -c $^ -o $@

clean:
	@rm -rf main main.o
	@rm -rf $(OBJDIR)

git:
	@$(MAKE) clean
	@git add .
	@git commit -m "$(MS)"
	@git push

runBlas:
	@$(MAKE) clean
	@$(MAKE) blas
	@mpirun -np $(NP) ./main $(SZ)

runCuda:
	@$(MAKE) clean
	@$(MAKE) cuda
	@mpirun -np $(NP) ./main $(SZ)
