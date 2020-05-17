CC=mpicc
CFLAGS= -O3 -std=c99
LIBS=-lmpi -lm

cg: main.c cg.c
	$(CC) $(CFLAGS) -g -Wformat=0 cg.c main.c -o cg $(LIBS)

clean: 
	rm -f cg
