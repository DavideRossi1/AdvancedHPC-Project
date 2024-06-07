CC=mpicc
INCLUDE=include
OBJDIR=obj
SRC=src
PRINTDIR=--no-print-directory

FLAGS=-O3 -I$(INCLUDE) 
OPT=-Wall -Wextra -march=native #-Wno-unknown-pragmas
OPENMP=-fopenmp
ACC=-acc -Minfo=all

OBJECTS=$(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))

#FLAGS+=-DDEBUG
FLAGS+=$(OPT)

cpu: FLAGS+=$(OPENMP)
cpu: jacobi.x

gpu: FLAGS+=$(ACC)
gpu: jacobi.x

jacobi.x: $(OBJECTS) jacobi.c
	@$(CC) $(FLAGS) $^ -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	@mkdir -p $(OBJDIR)
	@$(CC) $(FLAGS) -c $^ -o $@

clean:
	@rm -rf jacobi.x solution.dat $(OBJDIR)

plot:
	@gnuplot -p plot.plt
	
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

cpurun: FUNC=cpusave 
cpurun: run

gpurun: FUNC=gpusave
gpurun: run

run:
	@$(MAKE) clean $(PRINTDIR)
	@$(MAKE) $(FUNC) $(PRINTDIR)
	@mpirun -np $(NP) ./jacobi.x $(SZ) $(IT)

.PHONY: clean plot push pull run

