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

# small utility to check priority queue on Leonardo
prio:
	@echo "          JOBID PARTITION     USER   PRIORITY       SITE        AGE      ASSOC  FAIRSHARE    JOBSIZE  PARTITION        QOS        NICE                 TRES" > queue.txt
	@sprio -S -y -l | grep boost >> queue.txt

.PHONY: clean push pull prio