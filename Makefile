CC=mpicc
INCLUDE=include
OBJDIR=obj
SRC=src

FLAGS=-O3 -Wall -Wextra -march=native -I$(INCLUDE) -fopenmp
CBLAS=-DCBLAS -lopenblas

OBJECTS=$(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))


FLAGS+=-DDEBUG
#FLAGS+=-DPRINTTIME
FLAGS+=-DPRINTMATRIX

blas: FLAGS+=$(CBLAS)

blas: $(OBJECTS) main.c
	$(CC) $^ -o $@ $(FLAGS)
$(OBJDIR)/main.o: main.c
	$(CC) $(FLAGS) -c $^ -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	mkdir -p $(OBJDIR)
	$(CC) $(FLAGS) -c $^ -o $@

.PHONY: clean
clean:
	rm -rf blas main.o
	rm -rf $(OBJDIR)

runBlas:
	$(MAKE) clean
	$(MAKE) blas
	mpirun -np $(NP) ./blas $(SZ)
