CXX = g++
CXXFLAGS = -O3 -Wall -fopenmp -g -march=native -ffast-math -std=c++17
all: bb frb upchan
bb: bb.o
	$(CXX) $(CXXFLAGS) -o $@ $^
frb: frb.o
	$(CXX) $(CXXFLAGS) -o $@ $^
upchan: upchan.o
	$(CXX) $(CXXFLAGS) -o $@ $^
%.o: %.cxx
	$(CXX) $(CXXFLAGS) -c $*.cxx
format:
	clang-format -i bb.cxx frb.cxx upchan.cxx
clean:
	$(RM) bb bb.o
	$(RM) frb frb.o
	$(RM) upchan upchan.o
.PHONY: all format clean
