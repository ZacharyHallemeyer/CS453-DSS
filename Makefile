BIN=./dss
INCLUDE=./include
SRC=./src
TEST=./tests

all: tree $(SRC)/dss.cu
	$(CC) kd_tree.o $(SRC)/dss.cu -o $(BIN)

test: tree $(TEST)/test.c
	$(CC) kd_tree.o $(TEST)/test.c -o test
	./test

tree: $(INCLUDE)/kd_tree.h $(SRC)/kd_tree.c
	$(CC) -c $(SRC)/kd_tree.c

clean:
	rm -rf *.o $(BIN) ./test
