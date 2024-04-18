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
    float* dataset = (float*)malloc(sizeof(float) * N * DIM);
    struct kd_tree* tree;

    fill_dataset(dataset);

    init_kd_tree(&tree);

    for (unsigned int i = 0; i < N; i += 1)
    {
        struct kd_tree_node* node;
        float data[2];

        for (unsigned int d = 0; d < DIM; d += 1)
        {
            data[d] = dataset[i * DIM + d];
        }

        init_kd_tree_node(&node, data, DIM, 0);
        insert(&tree->head, &node);
    }

    print_tree(tree->head);

    free_kd_tree(&tree);
    
    return 0;
}


// function implementation
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
