AMREXPR_ROOT = ../../tmp_install_dir

CXX = g++
CXXFLAGS = -std=c++17 -O3 -g1 -I$(AMREXPR_ROOT)/include

default: a.out

a.out: main.o
	$(CXX) $(CXXFLAGS) $^ -L$(AMREXPR_ROOT)/lib -lamrexpr

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: clean

clean:
	$(RM) main.o a.out
