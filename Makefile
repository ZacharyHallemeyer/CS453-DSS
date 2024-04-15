BIN=./dss

all: tree dss.cu
	$(CC) kd_tree.o dss.cu -o $(BIN)

tree: kd_tree.h kd_tree.c
	$(CC) -c kd_tree.c

clean:
	rm -rf *.o $(BIN)
