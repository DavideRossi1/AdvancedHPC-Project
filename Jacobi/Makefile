CC = mpicc
INCLUDE = include
OBJDIR = obj
OUTPUTDIR = output
SRC = src
MAIN = jacobi.c
OBJECTS = $(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))

PRINTDIR = --no-print-directory
FLAGS = -O3 -I$(INCLUDE) -Wall -Wextra -march=native #-Wno-unknown-pragmas
OPENMP = -fopenmp
OPENACC = -acc #-Minfo=all

#FLAGS+=-DDEBUG

cpu: FLAGS += $(OPENMP)
cpu: jacobi.x

gpu: FLAGS += $(OPENACC)
gpu: jacobi.x

jacobi.x: $(OBJECTS) $(MAIN)
	@mkdir -p $(OUTPUTDIR)
	@$(CC) $(FLAGS) $^ -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	@mkdir -p $(OBJDIR)
	@$(CC) $(FLAGS) -c $^ -o $@

clean:
	@rm -rf jacobi.x $(OBJDIR) $(OUTPUTDIR)
	@$(MAKE) clean -C original_code $(PRINTDIR)

plot:
	@gnuplot -e "NITER=$(words $(wildcard output/*.dat))" -p plot.plt 

cpusave: FLAGS += -DSAVEPLOT
cpusave: cpu

gpusave: FLAGS += -DSAVEPLOT
gpusave: gpu

cpugif: FLAGS += -DSAVEGIF
cpugif: cpu

gpugif: FLAGS += -DSAVEGIF
gpugif: gpu

cpurun: FUNC = cpusave 
cpurun: run

gpurun: FUNC = gpusave
gpurun: run

run: clean 
	@$(MAKE) $(FUNC) $(PRINTDIR)
	@mpirun -np $(NP) ./jacobi.x $(SZ) $(IT)
	@$(MAKE) plot

comparecpu: F=cpurun
comparecpu: compare

comparegpu: F=gpurun
comparegpu: compare

compare:
	@$(MAKE) $(F) $(PRINTDIR) NP=$(NP) SZ=$(SZ) IT=$(IT) >> /dev/null
	@$(MAKE) run -C original_code $(PRINTDIR) dim=$(SZ) it=$(IT) r=1 c=1 >> /dev/null
	@diff ./$(OUTPUTDIR)/solution0.dat ./original_code/solution.dat || true

.PHONY: clean plot run cpu gpu
