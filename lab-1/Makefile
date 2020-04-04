CXX= g++
CXXFLAGS= -O2
LDFLAGS = -lpthread

all: shellsort shellsort-parallel radixsort radixsort-parallel

shellsort: main.cc Sorters.hh ShellSorter.cc HRTimer.hh
	$(CXX) $(CXXFLAGS) -DSHELL main.cc ShellSorter.cc -o shellsort $(LDFLAGS)

shellsort-parallel: main.cc Sorters.hh ShellSorter.cc HRTimer.hh
	$(CXX) $(CXXFLAGS) -DSHELL -DPARALLEL main.cc ShellSorter.cc -o shellsort-parallel $(LDFLAGS)

radixsort: main.cc Sorters.hh RadixSorter.cc HRTimer.hh
	$(CXX) $(CXXFLAGS) main.cc RadixSorter.cc -o radixsort $(LDFLAGS)

radixsort-parallel: main.cc Sorters.hh RadixSorter.cc HRTimer.hh
	$(CXX) $(CXXFLAGS) -DPARALLEL main.cc RadixSorter.cc -o radixsort-parallel $(LDFLAGS)

clean:
	rm -f shellsort
	rm -f shellsort-parallel
	rm -f radixsort
	rm -f radixsort-parallel
	rm -f *~
