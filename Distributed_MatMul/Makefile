CC=mpicc
INCLUDE=include
OBJDIR=obj
SRC=src
PRINTDIR=--no-print-directory

CFLAGS=-I$(INCLUDE) # compilation flags
MAINFLAGS=$(CFLAGS) # flags for main file
LINKFLAGS=$(CFLAGS) # flags for linking

OPT=-Wall -Wextra -march=native -O3
OPENMP=-fopenmp
CBLAS=-DCBLAS -lopenblas
CUDA=-DCUDA
COMPCAPABILITY=-arch=sm_80
MAIN=main.c
LDFLAGS=-lcublas -lcudart -lmpi -L/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/lib

OBJECTS=$(OBJDIR)/main.o $(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))

#CFLAGS+=-DDEBUG

naive: CFLAGS+=$(OPT) $(OPENMP)
naive: main

cpu: CFLAGS+=$(CBLAS) $(OPT) $(OPENMP)
cpu: main

gpu: CFLAGS+=$(CUDA)
gpu: MAINFLAGS+=$(COMPCAPABILITY)
gpu: LINKFLAGS=$(LDFLAGS)
gpu: CC=nvcc
gpu: MAIN=main.cu
gpu: prepare_main # for GPU computation, rename main.c to main.cu
gpu: main
gpu: cleanup_main # rename main.cu back to main.c

main: $(OBJECTS)
	$(CC) $(LINKFLAGS) $^ -o $@
$(OBJDIR)/main.o:
	@mkdir -p $(OBJDIR)
	$(CC) $(MAINFLAGS) -c $(MAIN) -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	mpicc $(CFLAGS) -c $< -o $@

prepare_main:
	@mv main.c main.cu
cleanup_main:
	@mv main.cu main.c

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

.PHONY: clean push pull run cpu gpu
