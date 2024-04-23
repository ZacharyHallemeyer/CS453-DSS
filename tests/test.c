// Used for testing on local systems before running on HPC cluster
#include <stdio.h>
#include <stdlib.h>

#include "../include/kd_tree.h"


// for testing with 3 points
#define N 10
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

    print_tree(tree->root);

    float query[2] = { 2.0, 0.0 };
    float epsilon = 3.0;
    unsigned int count = 0;
    points_within_epsilon(&tree, query, epsilon, &count);

    printf("\nfor point: { % 9.2f", query[0]);
    for (unsigned int i = 1; i < DIM; i += 1)
    {
        printf(", % 9.2f", query[i]);
    }
    printf(" }\n");
    printf("The number of points within epsilon = %f is %u\n", epsilon, count);

    free_kd_tree(&tree);
    
    return 0;
}


// function implementation
void fill_dataset(float* dataset)
{
    // fill first point
    dataset[0]  =   0.0;
    dataset[1]  =   0.0;
    // fill second point
    dataset[2]  =  -1.0;
    dataset[3]  =  -3.0;
    // fill third point
    dataset[4]  =   1.0;
    dataset[5]  =   0.0;
    // fill forth point
    dataset[6]  =  -2.0;
    dataset[7]  =   4.0;
    // fill fifth point
    dataset[8]  =  -5.0;
    dataset[9]  =  -1.0;
    // fill sixth point
    dataset[10] =  10.0;
    dataset[11] =   2.0;
    // fill seventh point
    dataset[12] =  -5.0;
    dataset[13] =  -6.0;
    // fill eighth point
    dataset[14] =  -3.0;
    dataset[15] = -10.0;
}
