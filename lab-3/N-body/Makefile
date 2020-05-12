CC=mpiCC
CFLAGS= -O3
LIBS=-lmpi -lm

nbody: body.cpp
	$(CC) $(CFLAGS) body.cpp -o nbody $(LIBS)

clean: 
	rm -f nbody
