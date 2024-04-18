#include <stdio.h>
#include <stdlib.h>

#include "../include/kd_tree.h"


// function implementation
void init_kd_tree(struct kd_tree** tree)
{
    *tree = (struct kd_tree*)malloc(sizeof(struct kd_tree));

    (*tree)->height = 0;
    (*tree)->head = NULL;
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
    (*node)->dim = dim;
    (*node)->data = (float*)malloc(sizeof(float) * dim);
    (*node)->left = NULL;
    (*node)->right = NULL;

    for (unsigned int d = 0; d < dim; d += 1)
    {
        (*node)->data[d] = data[d];
    }
}

void insert(struct kd_tree_node** node, struct kd_tree_node** new_node)
{
    if (*node == NULL)
    {
        *node = *new_node;
        (*new_node)->level = 0;
    }
    else if ((*new_node)->data[(*node)->level % (*node)->dim]
                 < (*node)->metric)
    {
        insert(&(*node)->left, new_node);
    }
    else if ((*new_node)->data[(*node)->level % (*node)->dim]
                 > (*node)->metric)
    {
        insert(&(*node)->right, new_node);
    }
    else
    {
        *node = *new_node;
        (*new_node)-> level = (*node)->level = 0;
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
        " level: %u, metric: %f, left: %d, right: %d,",
        node->level,
        node->metric,
        node->left,
        node->right
    );
    printf(" data: { %f", node->data[0]);
    for (unsigned int d = 0; d < node->dim; d += 1)
    {
        printf(", %f", node->data[d]);
    }
    printf(" }");
    printf(" },\n");

    if (node->right != NULL)
    {
        print_tree(node->right);
    }
}

void free_kd_tree(struct kd_tree** tree)
{
    __free_kd_tree(&(*tree)->head);

    free(*tree);
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
