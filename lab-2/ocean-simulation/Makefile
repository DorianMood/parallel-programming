
CXX = gcc
CXXFLAGS = -O3 -std=gnu11
LDFLAGS = 

all: serial_ocean omp_ocean

serial_ocean: serial_ocean.c ocean_main.c
	$(CXX) $(CXXFLAGS) ocean_main.c serial_ocean.c -o serial_ocean $(LDFLAGS)

omp_ocean: omp_ocean.c ocean_main.c
	$(CXX) $(CXXFLAGS) -DPARALLEL -fopenmp ocean_main.c omp_ocean.c -o omp_ocean $(LDFLAGS)

clean:
	rm -f serial_ocean
	rm -f omp_ocean
