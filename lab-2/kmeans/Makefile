CC = gcc
CC_FLAGS = -g -fopenmp -O3 -std=c99

all: kmeans-serial kmeans-parallel

kmeans-serial: main.c cluster.c kmeans_clustering.c kmeans.h
	${CC} ${CC_FLAGS} main.c cluster.c kmeans_clustering.c -o kmeans-serial

kmeans-parallel: main.c cluster.c kmeans_clustering.c kmeans.h
	${CC} ${CC_FLAGS} -DPARALLEL main.c cluster.c kmeans_clustering.c -o kmeans-parallel

clean:
	rm -f *~ kmeans-serial kmeans-parallel
