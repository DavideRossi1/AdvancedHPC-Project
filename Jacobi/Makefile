# inputs for the executables
CC=mpicc
INCLUDE=include
OBJDIR=obj
SRC=src
PRINTDIR=--no-print-directory

FLAGS=-O3 -I$(INCLUDE) -Wall -Wextra -march=native -Wno-unknown-pragmas

OBJECTS=$(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))

#FLAGS+=-DDEBUG

#FLAGS+=-acc -Minfo=all

jacobi.x: $(OBJECTS) jacobi.c
	@$(CC) $(FLAGS) $^ -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	@mkdir -p $(OBJDIR)
	@$(CC) $(FLAGS) -c $^ -o $@

plot:
	@gnuplot -p plot.plt
	
clean:
	@rm -rf jacobi.x solution.dat $(OBJDIR)

push:
	@$(MAKE) clean $(PRINTDIR)
	@git add .
	@git commit -m "$(MS)"
	@git push
pull:
	@git reset --hard
	@git pull

run:
	@$(MAKE) clean $(PRINTDIR)
	@$(MAKE) $(PRINTDIR)
	@mpirun -np $(NP) ./jacobi.x $(SZ) $(IT)

.PHONY: clean plot push pull

