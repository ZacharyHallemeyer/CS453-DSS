#include <stdio.h>
#include <stdlib.h>

#include "../include/kd_tree.h"


// function implementation
void init_kd_tree(
        struct kd_tree* tree,
        const float* data,
        const unsigned int DIM
) {
    struct kd_tree_node* head =
        (struct kd_tree_node*)malloc(sizeof(struct kd_tree_node));
    init_kd_tree_node(head, data, DIM, 0);

    tree->height = 0;
    tree->head = head;
}

void init_kd_tree_node(
        struct kd_tree_node* node,
        const float* data,
        const unsigned int DIM,
        const unsigned int level
) {
    // TODO: allocate space on heap here
    node->level = level;
    node->metric = data[level % DIM];
    // TODO: add data for point
    node->left = NULL;
    node->right = NULL;
}

void insert(
        struct kd_tree* tree,
        const float* data,
        const unsigned int dim
) {
    struct kd_tree_node* working_node = tree->head;
    struct kd_tree_node* parent_node = NULL;

    while (working_node != NULL)
    {
        if (working_node->metric < data[working_node->level % dim])
        {
            parent_node = working_node;
            working_node = working_node->left;
        }
        else if (working_node->metric > data[working_node->level % dim])
        {
            parent_node = working_node;
            working_node = working_node->right;
        }
        // otherwise, equals?
    }

    working_node = (struct kd_tree*)malloc(sizeof(struct kd_tree));
    init_kd_tree_node(working_node, data, dim, working->level + 1);
}

void show(struct kd_tree* tree)
{
    struct kd_tree_node* working_node = tree->head;

    // TODO: implement tree traversal
    printf(
        // TODO: change 'left' and 'right' to be the addresses of the children
        "{ level: %u, metric: %f, left: %d, right: %d }\n",
        working_node->level,
        working_node->metric,
        working_node->left,
        working_node->right
    );
}
