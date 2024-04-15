BIN=./dss
INCLUDE=./include
SRC=./src

all: tree $(SRC)/dss.cu
	$(CC) kd_tree.o $(SRC)/dss.cu -o $(BIN)

tree: $(INCLUDE)/kd_tree.h $(SRC)/kd_tree.c
	$(CC) -c $(SRC)/kd_tree.c

clean:
	rm -rf *.o $(BIN)
