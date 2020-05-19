CC=mpiCC
CFLAGS= -O3
LIBS=-lmpi -lm

main: main.cpp
	$(CC) $(CFLAGS) -g -Wformat=0 main.cpp -o a.out $(LIBS)

clean: 
	rm -f a.out
