# inputs for the executables
CC=mpicc
INCLUDE = include
OBJDIR=obj
SRC=src

FLAGS=-O3 -I$(INCLUDE) -Wall -Wextra -march=native -g #-Wno-unknown-pragmas

OBJECTS=$(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))

#FLAGS+=-DDEBUG
#FLAGS+=-DCONVERT
FLAGS+=-DPRINTTIME

FLAGS+=-acc -Minfo=all

jacobi.x: $(OBJECTS) jacobi.c
	@$(CC) $(FLAGS) $^ -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	@mkdir -p $(OBJDIR)
	@$(CC) $(FLAGS) -c $^ -o $@

plot:
	@gnuplot -p plot.plt
	
clean:
	@rm -rf jacobi.x solution.*
	@rm -rf $(OBJDIR)

push:
	@$(MAKE) clean $(PRINTDIR)
	@git add .
	@git commit -m "$(MS)"
	@git push
pull:
	@git reset --hard
	@git pull

.PHONY: clean plot push pull

