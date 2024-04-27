CC=mpicc
INCLUDE=include
OBJDIR=obj
SRC=src

FLAGS=-O3 -Wall -Wextra -march=native -I$(INCLUDE)
OPENMP=-fopenmp
CBLAS=-DCBLAS -lopenblas
CUDA=-DCUDA -lcublas -lcudart

OBJECTS=$(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))

FLAGS+=$(OPENMP)
#FLAGS+=-DDEBUG
#FLAGS+=-DPRINTTIME
#FLAGS+=-DPRINTMATRIX

main: $(OBJECTS) main.c
	$(CC) $(FLAGS) $^ -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(FLAGS) -c $^ -o $@

blas: FLAGS+=$(CBLAS)
blas: main
	@$(MAKE) main --no-print-directory

cuda: FLAGS+=$(CUDA)
cuda: main
	$(MAKE) main --no-print-directory


clean:
	@rm -rf main main.o
	@rm -rf $(OBJDIR)

runBlas:
	@$(MAKE) clean
	@$(MAKE) blas
	@mpirun -np $(NP) ./main $(SZ)

runCuda:
	@$(MAKE) clean
	@$(MAKE) cuda
	@mpirun -np $(NP) ./main $(SZ)
