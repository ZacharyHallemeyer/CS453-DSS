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
    node->left = NULL;
    node->right = NULL;
}

void insert_data(struct kd_tree* tree, const float* data)
{
}
