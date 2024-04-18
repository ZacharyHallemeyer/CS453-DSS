// Used for testing on local systems before running on HPC cluster
#include <stdio.h>
#include <stdlib.h>

#include "../include/kd_tree.h"


// for testing with 3 points
#define N 4
// for testing with points in 2 dimensions
#define DIM 2


// function prototypes
void fill_dataset(float* dataset);


int main(int argc, char** argv)
{
    float* dataset = (float*)malloc(sizeof(float) * N * DIM);

    struct kd_tree* tree;
    // struct kd_tree_node* head;
    // struct kd_tree_node* node;

    fill_dataset(dataset);

    // float head_data[2] = { dataset[0], dataset[1] };
    // float node_data[2] = { dataset[2], dataset[3] };

    init_kd_tree(&tree);
    // init_kd_tree_node(&head, head_data, DIM, 0);
    // init_kd_tree_node(&node, node_data, DIM, 0);
    // insert(&tree, &head);
    // insert(&tree, &node);

    for (unsigned int p = 0; p < N; p += 1)
    {
        struct kd_tree_node* node;
        float data[2];

        for (unsigned int d = 0; d < DIM; d += 1)
        {
            data[d] = dataset[p * DIM + d];
        }

        init_kd_tree_node(&node, data, DIM, 0);
        insert(&tree, &node);
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
    // fill forth point
    dataset[6] = -20.0;
    dataset[7] = -10.0;
}
