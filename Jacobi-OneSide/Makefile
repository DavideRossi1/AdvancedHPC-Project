CC = mpicc
INCLUDE = include
OBJDIR = obj
OUTPUTDIR = output
SRC = src
MAIN = main.c
OBJECTS = $(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))

PRINTDIR = --no-print-directory
FLAGS = -O3 -I$(INCLUDE) -Wall -Wextra -march=native -fopenmp #-Wno-unknown-pragmas

#FLAGS+=-DDEBUG

main: $(OBJECTS) $(MAIN)
	@mkdir -p $(OUTPUTDIR)
	@$(CC) $(FLAGS) $^ -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	@mkdir -p $(OBJDIR)
	@$(CC) $(FLAGS) -c $^ -o $@

clean:
	@rm -rf main $(OBJDIR) $(OUTPUTDIR)
	@$(MAKE) clean -C original_code $(PRINTDIR)

plot:
	@gnuplot -e "NITER=$(words $(wildcard output/*.dat))" -p plot.plt 

save: FLAGS+=-DSAVEPLOT
save: main

gif: FLAGS+=-DSAVEGIF
gif: main

run:
	@$(MAKE) clean $(PRINTDIR)
	@$(MAKE) save $(PRINTDIR)
	@mpirun -np $(NP) ./main $(SZ) $(IT)

compare:
	@$(MAKE) run $(PRINTDIR) NP=$(NP) SZ=$(SZ) IT=$(IT) >> /dev/null
	@$(MAKE) run -C original_code $(PRINTDIR) dim=$(SZ) it=$(IT) r=1 c=1 >> /dev/null
	@diff ./$(OUTPUTDIR)/solution0.dat ./original_code/solution.dat || true

.PHONY: clean plot run cpu gpu

