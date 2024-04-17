#include <stdio.h>
#include <stdlib.h>

#include "../include/kd_tree.h"


// function implementation
void init_kd_tree(
        struct kd_tree* tree,
        const float* data,
        const unsigned int dim
) {
    struct kd_tree_node* head = NULL;
    init_kd_tree_node(&head, data, dim, 0);

    tree->height = 0;
    tree->head = head;
}

void init_kd_tree_node(
        struct kd_tree_node** node,
        const float* data,
        const unsigned int dim,
        const unsigned int level
) {
    *node = (struct kd_tree_node*)malloc(sizeof(struct kd_tree_node));
    (*node)->level = level;
    (*node)->metric = data[level % dim];
    // TODO: add data for point
    (*node)->left = NULL;
    (*node)->right = NULL;
}

void insert(
        struct kd_tree_node** node,
        const float* data,
        const unsigned int p,
        const unsigned int dim,
        const unsigned int level
) {
    unsigned int p_idx = p * dim + ((*node)->level % dim);
    
    if (*node == NULL)
    {
        init_kd_tree_node(node, data, dim, (*node)->level + 1);
    }
    if (data[p_idx] < (*node)->metric)
    {
        insert(&(*node)->left, data, p, dim, (*node)->level + 1);
    }
    else if (data[p_idx] > (*node)->metric)
    {
        insert(&(*node)->right, data, p, dim, (*node)->level + 1);
    }
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
