CC=mpicc
INCLUDE=include
OBJDIR=obj
OUTPUTDIR=output
SRC=src
PRINTDIR=--no-print-directory

FLAGS=-O3 -I$(INCLUDE) -Wall -Wextra -march=native #-Wno-unknown-pragmas
OPENMP=-fopenmp
OPENACC=-acc #-Minfo=all

OBJECTS=$(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))

#FLAGS+=-DDEBUG

cpu: FLAGS+=$(OPENMP)
cpu: jacobi.x

gpu: FLAGS+=$(OPENACC)
gpu: jacobi.x

jacobi.x: $(OBJECTS) jacobi.c
	@mkdir -p $(OBJDIR)
	@mkdir -p $(OUTPUTDIR)
	@$(CC) $(FLAGS) $^ -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	@$(CC) $(FLAGS) -c $^ -o $@

clean:
	@rm -rf jacobi.x solution.* $(OBJDIR) $(OUTPUTDIR)

plot:
	@gnuplot -e "NITER=$(words $(wildcard output/*.dat))" -p plot.plt 
	
push:
	@$(MAKE) clean $(PRINTDIR)
	@git add .
	@git commit -m "$(MS)"
	@git push
pull:
	@git reset --hard
	@git pull

cpusave: FLAGS+=-DSAVEPLOT
cpusave: cpu

gpusave: FLAGS+=-DSAVEPLOT
gpusave: gpu

cpugif: FLAGS+=-DSAVEGIF
cpugif: cpu

gpugif: FLAGS+=-DSAVEGIF
gpugif: gpu

cpurun: FUNC=cpusave 
cpurun: run

gpurun: FUNC=gpusave
gpurun: run

run:
	@$(MAKE) clean $(PRINTDIR)
	@$(MAKE) $(FUNC) $(PRINTDIR)
	@mpirun -np $(NP) ./jacobi.x $(SZ) $(IT)

comparecpu: F=cpurun
comparecpu: compare

comparegpu: F=gpurun
comparegpu: compare

compare:
	@$(MAKE) clean $(PRINTDIR)
	@$(MAKE) $(F) $(PRINTDIR) NP=$(NP) SZ=$(SZ) IT=$(IT) >> /dev/null
	@$(MAKE) plot $(PRINTDIR)
	@$(MAKE) run -C original_code $(PRINTDIR) dim=$(SZ) it=$(IT) r=1 c=1 >> /dev/null
	@$(MAKE) plot -C original_code $(PRINTDIR)
	@diff ./solution.png ./original_code/solution.png || true

.PHONY: clean plot push pull run cpu gpu diff

