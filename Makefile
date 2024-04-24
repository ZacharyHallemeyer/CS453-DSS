BIN=./dss
INCLUDE=./include
SRC=./src
TEST=./tests

all: tree $(SRC)/DSS.cu
	$(CC) kd_tree.o $(SRC)/DSS.cu -o $(BIN)

test: tree $(TEST)/test.c
	$(CC) kd_tree.o $(TEST)/test.c -o test
	./test
	rm -rf ./test

tree: $(INCLUDE)/kd_tree.h $(SRC)/kd_tree.c
	$(CC) -c $(SRC)/kd_tree.c

clean:
	rm -rf *.o $(BIN)
