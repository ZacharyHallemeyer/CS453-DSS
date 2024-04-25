BIN=./DSS
INCLUDE=./include
SRC=./src
TEST=./tests
ARCH=80

all: tree $(SRC)/DSS.cu
	nvcc -O3 -arch=compute_$(ARCH) -code=sm_$(ARCH) -lcuda -lineinfo -Xcompiler -fopenmp $(SRC)/DSS.cu -o $(BIN) -lm

test: tree $(TEST)/test.c
	$(CC) kd_tree.o $(TEST)/test.c -o test -lm
	./test
	rm -rf ./test

tree: $(INCLUDE)/kd_tree.h $(SRC)/kd_tree.c
	$(CC) -c $(SRC)/kd_tree.c -lm

clean:
	rm -rf *.o $(BIN)
