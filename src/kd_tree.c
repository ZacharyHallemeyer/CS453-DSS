#include <stdio.h>
#include <stdlib.h>

#include "../include/kd_tree.h"


// function implementation
void __points_within_epsilon(
    struct kd_tree_node_cpu** node,
    const float* query,
    const float epsilon,
    unsigned int* count
) {
    if (*node == NULL)
    {
        return;
    }


    float dist = 0.0;
    struct kd_tree_node_cpu** first_node = NULL;
    struct kd_tree_node_cpu** second_node = NULL;


    if (query[(*node)->level % (*node)->dim]
            < (*node)->data[(*node)->level % (*node)->dim])
    {
        first_node = &(*node)->left;
        second_node = &(*node)->right;
    }
    else
    {
        first_node = &(*node)->right;
        second_node = &(*node)->left;
    }

    __points_within_epsilon(first_node, query, epsilon, count);

    for (unsigned int i = 0; i < (*node)->dim; i += 1)
    {
        dist += (query[i] - (*node)->data[i])
                    * (query[i] - (*node)->data[i]);
    }
    dist = sqrt(dist);

    if ((*node)->data[(*node)->level % (*node)->dim] <= epsilon)
    {
        __points_within_epsilon(second_node, query, epsilon, count);
    }

    if (dist <= epsilon)
    {
        *count += 1;
    }
}


void __free_kd_tree_cpu(struct kd_tree_node_cpu** node)
{
    if (*node == NULL)
    {
        return;
    }

    if ((*node)->left != NULL)
    {
        __free_kd_tree_cpu(&(*node)->left);
    }

    if ((*node)->right != NULL)
    {
        __free_kd_tree_cpu(&(*node)->right);
    }

    free(*node);
}


void __insert(
    struct kd_tree_node_cpu** parent,
    struct kd_tree_node_cpu** node,
    struct kd_tree_node_cpu** new_node,
    const unsigned int level
) {
    if (*node == NULL)
    {
        *node = *new_node;
        (*new_node)->level = level + 1;
        (*new_node)->metric =
            (*new_node)->data[(*new_node)->level % (*new_node)->dim];
        (*new_node)->parent = *parent;
    }
    else if ((*new_node)->data[level % (*new_node)->dim] < (*node)->metric)
    {
        __insert(node, &(*node)->left, new_node, (*node)->level);
    }
    else if ((*new_node)->data[level % (*new_node)->dim] > (*node)->metric)
    {
        __insert(node, &(*node)->right, new_node, (*node)->level);
    }
}


void points_within_epsilon(
    struct kd_tree_cpu** tree,
    const float* query,
    const float epsilon,
    unsigned int* count
) {
    if ((*tree)->root != NULL)
    {
        __points_within_epsilon(&(*tree)->root, query, epsilon, count);
    }
}


void free_kd_tree_cpu(struct kd_tree_cpu** tree)
{
    __free_kd_tree_cpu(&(*tree)->root);

    free(*tree);
}


void init_kd_tree_cpu(struct kd_tree_cpu** tree)
{
    *tree = (struct kd_tree_cpu*)malloc(sizeof(struct kd_tree_cpu));

    (*tree)->height = 0;
    (*tree)->root = NULL;
}


void init_kd_tree_node_cpu(
        struct kd_tree_node_cpu** node,
        const float* data,
        const unsigned int dim,
        const unsigned int level
) {
    *node = (struct kd_tree_node_cpu*)malloc(sizeof(struct kd_tree_node_cpu));
    (*node)->level = level;
    (*node)->metric = 0;
    (*node)->dim = dim;
    (*node)->data = (float*)malloc(sizeof(float) * dim);
    (*node)->left = NULL;
    (*node)->right = NULL;
    (*node)->parent = NULL;

    for (unsigned int d = 0; d < dim; d += 1)
    {
        (*node)->data[d] = data[d];
    }
}


void insert(struct kd_tree_cpu** tree, struct kd_tree_node_cpu** new_node)
{
    if ((*tree)->root == NULL)
    {
        (*tree)->root = *new_node;
        (*new_node)->level = 0;
        (*new_node)->metric =
            (*new_node)->data[(*new_node)->level % (*new_node)->dim];
        (*new_node)->parent = NULL;
    }
    else
    {
        __insert(&(*tree)->root, &(*tree)->root, new_node, 0);
    }
}


void print_tree(struct kd_tree_node_cpu* node)
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
        " level: %5u, metric: % 9.2f, left: %9d, right: %9d, parent: %9d",
        node->level,
        node->metric,
        node->left,
        node->right,
        node->parent
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
