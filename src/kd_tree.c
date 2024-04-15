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
    node->level = level;
    node->metric = data[level % DIM];
    // TODO: add data for point
    node->left = NULL;
    node->right = NULL;
}

void insert_data(struct kd_tree* tree, const float* data)
{
}

void show(struct kd_tree* tree)
{
    struct kd_tree_node* working_node = tree->head;

    printf(
        "{ level: %u, metric: %f, left: %d, right: %d }\n",
        working_node->level,
        working_node->metric,
        working_node->left,
        working_node->right
    );

    // TODO: implement tree traversal
}
