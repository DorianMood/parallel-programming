CC=gcc
CXX=g++
NVCC=nvcc

NVCCFLAGS= -O3 -std=c++11

knn-exec: main.cu knnCPU.cpp
	$(NVCC) $(NVCCFLAGS) main.cu knnCPU.cpp -o knn-exec

clean:
	rm -f knn-exec
