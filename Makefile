PRINTDIR=--no-print-directory

clean:
	@$(MAKE) clean -C Distributed_MatMul $(PRINTDIR)
	@$(MAKE) clean -C Jacobi $(PRINTDIR)
	@$(MAKE) clean -C Jacobi-OneSide $(PRINTDIR)

push:
	@$(MAKE) clean $(PRINTDIR)
	@git add .
	@git commit -m "$(MS)"
	@git push

pull:
	@git reset --hard
	@git pull

.PHONY: clean push pull