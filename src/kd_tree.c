#include <stdio.h>
#include <stdlib.h>

#include "../include/kd_tree.h"


// function implementation
void __closest_point(
    struct kd_tree_node** node,
    const float* data,
    const unsigned int dim
) {
}


void __free_kd_tree(struct kd_tree_node** node)
{
    if (*node == NULL)
    {
        return;
    }

    if ((*node)->left != NULL)
    {
        __free_kd_tree(&(*node)->left);
    }

    if ((*node)->right != NULL)
    {
        __free_kd_tree(&(*node)->right);
    }

    free(*node);
}


void __insert(
    struct kd_tree_node** node,
    struct kd_tree_node** new_node,
    const unsigned int level
) {
    if (*node == NULL)
    {
        *node = *new_node;
        (*new_node)->level = level + 1;
        (*new_node)->metric =
            (*new_node)->data[(*new_node)->level % (*new_node)->dim];
    }
    else if ((*new_node)->data[level % (*new_node)->dim] < (*node)->metric)
    {
        __insert(&(*node)->left, new_node, (*node)->level);
    }
    else if ((*new_node)->data[level % (*new_node)->dim] > (*node)->metric)
    {
        __insert(&(*node)->right, new_node, (*node)->level);
    }
}


void closest_point(struct kd_tree** tree, float* data, const unsigned int dim)
{
}


void free_kd_tree(struct kd_tree** tree)
{
    __free_kd_tree(&(*tree)->root);

    free(*tree);
}


void init_kd_tree(struct kd_tree** tree)
{
    *tree = (struct kd_tree*)malloc(sizeof(struct kd_tree));

    (*tree)->height = 0;
    (*tree)->root = NULL;
}


void init_kd_tree_node(
        struct kd_tree_node** node,
        const float* data,
        const unsigned int dim,
        const unsigned int level
) {
    *node = (struct kd_tree_node*)malloc(sizeof(struct kd_tree_node));
    (*node)->level = level;
    (*node)->metric = 0;
    (*node)->dim = dim;
    (*node)->data = (float*)malloc(sizeof(float) * dim);
    (*node)->left = NULL;
    (*node)->right = NULL;

    for (unsigned int d = 0; d < dim; d += 1)
    {
        (*node)->data[d] = data[d];
    }
}


void insert(struct kd_tree** tree, struct kd_tree_node** new_node)
{
    if ((*tree)->root == NULL)
    {
        (*tree)->root = *new_node;
        (*new_node)->level = 0;
        (*new_node)->metric =
            (*new_node)->data[(*new_node)->level % (*new_node)->dim];
    }
    else
    {
        __insert(&(*tree)->root, new_node, 0);
    }
}


void print_tree(struct kd_tree_node* node)
{
    if (node == NULL)
    {
        return;
    }

    if (node->left != NULL)
    {
        print_tree(node->left);
    }

    printf("{");
    printf(
        " level: %5u, metric: % 9.2f, left: %9d, right: %9d,",
        node->level,
        node->metric,
        node->left,
        node->right
    );
    printf(" data: { % 9.2f", node->data[0]);
    for (unsigned int d = 1; d < node->dim; d += 1)
    {
        printf(", % 9.2f", node->data[d]);
    }
    printf(" }");
    printf(" },\n");

    if (node->right != NULL)
    {
        print_tree(node->right);
    }
}
