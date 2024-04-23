CC=mpicc
INCLUDE=include

CFLAGS=-O3 -Wall -Wextra -march=native -I$(INCLUDE) -lblas
OBJDIR=obj
SRC=src
OBJECTS=$(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(wildcard $(SRC)/*.c))

main: $(OBJECTS) main.c
	$(CC) $^ -o $@ $(CFLAGS)
$(OBJDIR)/main.o: main.c
	$(CC) $(CFLAGS) -c $^ -o $@
$(OBJDIR)/%.o: $(SRC)/%.c
	mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $^ -o $@
clean:
	rm -rf main
	rm -rf $(OBJDIR)
