// Used for testing on local systems before running on HPC cluster
#include <stdio.h>
#include <stdlib.h>

#include "../include/kd_tree.h"


// for testing with 3 points
#define N 3
// for testing with points in 2 dimensions
#define DIM 2

// function prototypes
void fill_dataset(float* dataset);


int main(int argc, char** argv)
{
    struct kd_tree* tree = (struct kd_tree*)malloc(sizeof(struct kd_tree));
    float* dataset = (float*)malloc(sizeof(float) * N * DIM);

    fill_dataset(dataset);

    init_kd_tree(tree, dataset, DIM);
    show(tree);
    
    return 0;
}


void fill_dataset(float* dataset)
{
    // fill first point
    dataset[0] = 0.0;
    dataset[1] = 0.0;
    // fill second point
    dataset[2] = -1.0;
    dataset[3] = 0.0;
    // fill third point
    dataset[4] = 1.0;
    dataset[5] = 0.0;
}
