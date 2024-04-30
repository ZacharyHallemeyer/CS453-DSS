#include <stdio.h>
#include <stdlib.h>

#include "../include/kd_tree.cuh"


// function implementation
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
    else if ((*new_node)->data[(*node)->level % (*new_node)->dim] < (*node)->metric)
    {
        __insert(node, &(*node)->left, new_node, (*node)->level);
    }
    else if ((*new_node)->data[(*node)->level % (*new_node)->dim] > (*node)->metric)
    {
        __insert(node, &(*node)->right, new_node, (*node)->level);
    }
}


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
    float dist_prime = 0.0;
    struct kd_tree_node_cpu** first_node = NULL;
    struct kd_tree_node_cpu** second_node = NULL;


    for (unsigned int i = 0; i < (*node)->dim; i += 1)
    {
        dist += (query[i] - (*node)->data[i])
                    * (query[i] - (*node)->data[i]);
    }
    dist = sqrt(dist);

    dist_prime = fabsf(query[(*node)->level % (*node)->dim] - (*node)->metric);

    if (query[(*node)->level % (*node)->dim] < (*node)->metric)  // TODO: changed to using `metric` instead of refetching the dimension
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

    if (dist_prime < epsilon)
    {
        __points_within_epsilon(second_node, query, epsilon, count);
    }

    if (dist <= epsilon)
    {
        *count += 1;
    }
}


void __print_tree(struct kd_tree_node_cpu* node)
{
    if (node == NULL)
    {
        return;
    }

    if (node->left != NULL)
    {
        __print_tree(node->left);
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
        __print_tree(node->right);
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

    (*tree)->size = 0;
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
        (*tree)->size += 1;
    }
    else
    {
        __insert(&(*tree)->root, &(*tree)->root, new_node, 0);
        (*tree)->size += 1;
    }
}


void print_tree(struct kd_tree_cpu* tree)
{
    printf("\nNodes in tree: %d\n", tree->size);
    __print_tree(tree->root);
}


void __print_tree(struct kd_tree_node_cpu* node)
{
    if (node == NULL)
    {
        return;
    }

    if (node->left != NULL)
    {
        __print_tree(node->left);
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
        __print_tree(node->right);
    }
}


// ============== GPU


void allocate_gpu_memory(struct kd_tree_node_cpu** cpu_nodes, struct kd_tree_node_gpu** gpu_nodes, int num_nodes) {

    printf("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH\n\n\n");
    
    printf("%f\n", (*cpu_nodes)->metric);
    
    return;

  cudaMalloc((void**)&(*gpu_nodes), num_nodes * sizeof(struct kd_tree_node_gpu));

  // Copy each node
  for (int i = 0; i < num_nodes; i++)
  {
  
    printf("%f\n", cpu_nodes[i]->metric);
    struct kd_tree_node_gpu gpu_node;
    gpu_node.level = cpu_nodes[i]->level;
    gpu_node.metric = cpu_nodes[i]->metric;
    gpu_node.dim = cpu_nodes[i]->dim;
    gpu_node.left_child_index = -1;
    gpu_node.right_child_index = -1;
    gpu_node.parent_index = -1;


    if( cpu_nodes[i]->left )
    {
        gpu_node.left_child_index = 2 * i + 1;
    }

    if( cpu_nodes[i]->right )
    {
        gpu_node.right_child_index = 2 * i + 2;
    }
    if( cpu_nodes[i]->parent )
    {
        for( int j = 0; j < num_nodes; j++)
        {
            // Check if current node is parent node
            if( cpu_nodes[j] == cpu_nodes[i]->parent)
            {
                gpu_node.parent_index = j;
            }
        }
    }


    //cudaMalloc((void**)&(gpu_node.data), gpu_node.dim * sizeof(float));
    //cudaMemcpy(gpu_node.data, cpu_nodes[i]->data, gpu_node.dim * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy( &((*gpu_nodes)[i]) , &gpu_node, sizeof(struct kd_tree_node_gpu), cudaMemcpyHostToDevice);
    }
}
