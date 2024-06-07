CC=mpicc
INCLUDE=include
OBJDIR=obj
SRC=src
PRINTDIR=--no-print-directory

FLAGS=-O3 -I$(INCLUDE)
OPT=-Wall -Wextra -march=native
OPENMP=-fopenmp
CBLAS=-DCBLAS -lopenblas
CUDA=-DCUDA -lcublas -lcudart -lmpi -L/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/lib

OBJECTS=$(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))

#FLAGS+=-DDEBUG

naive: FLAGS+=$(OPT) $(OPENMP)
naive: main

cpu: FLAGS+=$(CBLAS) $(OPT) $(OPENMP)
cpu: main

gpu: FLAGS+=$(CUDA)
gpu: CC=nvcc
gpu: main

main: $(OBJECTS) main.c
	@$(CC) $(FLAGS) $^ -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	@mkdir -p $(OBJDIR)
	@$(CC) $(FLAGS) -c $^ -o $@

clean:
	@rm -rf main $(OBJDIR)

push:
	@$(MAKE) clean $(PRINTDIR)
	@git add .
	@git commit -m "$(MS)"
	@git push
pull:
	@git reset --hard
	@git pull

naiverun: FUNC=naive
naiverun: run

cpurun: FUNC=cpu
cpurun: run

gpurun: FUNC=gpu
gpurun: run
	
run:
	@$(MAKE) clean $(PRINTDIR)
	@$(MAKE) $(FUNC) $(PRINTDIR)
	@mpirun -np $(NP) ./main $(SZ)

.PHONY: clean push pull run
