BIN=./DSS
INCLUDE=./include
SRC=./src
TEST=./tests
ARCH=80

all: $(INCLUDE)/kd_tree.cuh $(SRC)/kd_tree.cu $(SRC)/DSS.cu
	nvcc -O3 -DMODE=1 -arch=compute_$(ARCH) -code=sm_$(ARCH) -lcuda -lineinfo -Xcompiler -fopenmp $(SRC)/kd_tree.cu $(SRC)/DSS.cu -o $(BIN)

test: tree $(TEST)/test.c
	$(CC) kd_tree.o -o test -lm
	./test
	rm -rf ./test

tree: $(INCLUDE)/kd_tree.h $(SRC)/kd_tree.c
	$(CC) -c $(SRC)/kd_tree.c -lm

clean:
	rm -rf *.o $(BIN)
