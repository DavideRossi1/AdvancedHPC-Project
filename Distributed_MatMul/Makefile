CC = mpicc
INCLUDE = include
OBJDIR = obj
SRC = src
MAIN = main.c
OBJECTS = $(OBJDIR)/main.o $(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))

PRINTDIR = --no-print-directory
OPT = -Wall -Wextra -march=native -O3
GPUOPT = -arch=sm_80 -O3
OPENMP = -fopenmp
CBLAS = -DCBLAS -lopenblas -lblas
CUDA = -DCUDA

CFLAGS = -I$(INCLUDE) $(OPENMP) $(OPT)# compilation flags
MAINFLAGS = -I$(INCLUDE) # flags for main file
LINKFLAGS = $(OPENMP) $(OPT) # flags for linking
GPULINKFLAGS = -lcublas -lcudart -lmpi -lnvomp -L/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/lib -O3

#CFLAGS+=-DDEBUG

naive: MAINFLAGS += $(OPT)
naive: main

cpu: MAINFLAGS += $(CBLAS) $(OPT)
cpu: LINKFLAGS += $(CBLAS)
cpu: main

gpu: MAINFLAGS += $(CUDA) $(GPUOPT)
gpu: LINKFLAGS = $(GPULINKFLAGS)
gpu: CC = nvcc
gpu: MAIN = main.cu
gpu: prepare_main  main  cleanup_main # for GPU computation, rename main.c to main.cu before compilation, and rename it back after compilation

main: $(OBJECTS)
	@$(CC) $(LINKFLAGS) $^ -o $@
$(OBJDIR)/main.o:
	@mkdir -p $(OBJDIR)
	@$(CC) $(MAINFLAGS) -c $(MAIN) -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	@mpicc $(CFLAGS) -c $< -o $@

prepare_main:
	@mv main.c main.cu
cleanup_main:
	@mv main.cu main.c

clean:
	@rm -rf main $(OBJDIR)

naiverun: FUNC=naive
naiverun: run

cpurun: FUNC=cpu
cpurun: run

gpurun: FUNC=gpu
gpurun: run
	
run: clean
	@$(MAKE) $(FUNC) $(PRINTDIR)
	@mpirun -np $(NP) ./main $(SZ)

.PHONY: clean run cpu gpu
