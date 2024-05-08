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
    struct kd_tree_node_cpu** new_node
) {
    if (*node == NULL)
    {
        *node = *new_node;
        (*new_node)->level = (*parent)->level + 1;
        (*new_node)->metric =
            (*new_node)->data[(*new_node)->level % (*new_node)->dim];
        (*new_node)->parent = *parent;
    }
    else if ((*new_node)->data[(*node)->level % (*new_node)->dim] < (*node)->metric)
    {
        __insert(node, &(*node)->left, new_node);
    }
    else if ((*new_node)->data[(*node)->level % (*new_node)->dim] > (*node)->metric)
    {
        __insert(node, &(*node)->right, new_node);
    }
}


void __points_within_epsilon_cpu(
        struct kd_tree_node_cpu** node,
        double* query,
        const double epsilon,
        unsigned int* count
) {
    if (*node == NULL)
    {
        return;
    }


    double dist = 0.0;
    double dist_prime = 0.0;
    struct kd_tree_node_cpu** first = NULL;
    struct kd_tree_node_cpu** second = NULL;

    for (unsigned int i = 0; i < (*node)->dim; i += 1)
    {
        dist += (query[i] - (*node)->data[i])
                    * (query[i] - (*node)->data[i]);
    }
    dist = sqrt(dist);

    dist_prime = fabsf(query[(*node)->level % (*node)->dim] - (*node)->metric);

    if (query[(*node)->level % (*node)->dim] < (*node)->metric)
    {
        first = &(*node)->left;
        second = &(*node)->right;
    }
    else
    {
        first = &(*node)->right;
        second = &(*node)->left;
    }

    __points_within_epsilon_cpu(first, query, epsilon, count);

    if (dist_prime < epsilon)
    {
        __points_within_epsilon_cpu(second, query, epsilon, count);
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


void points_within_epsilon_cpu(
    struct kd_tree_cpu** tree,
    double* query,
    const double epsilon,
    unsigned int* count
) {
    if ((*tree)->root != NULL)
    {
        __points_within_epsilon_cpu(&(*tree)->root, query, epsilon, count);
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
        const double* data,
        const unsigned int dim
) {
    *node = (struct kd_tree_node_cpu*)malloc(sizeof(struct kd_tree_node_cpu));
    (*node)->level = 0;
    (*node)->metric = 0;
    (*node)->dim = dim;
    (*node)->data = (double*)malloc(sizeof(double) * dim);
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
        __insert(&(*tree)->root, &(*tree)->root, new_node);
        (*tree)->size += 1;
    }
}


void print_tree(struct kd_tree_cpu* tree)
{
    printf("\nNodes in tree: %d\n", tree->size);
    __print_tree(tree->root);
}

void set_tree_height(struct kd_tree_node_cpu* node, unsigned int* max_height)
{
    if(node != NULL)
    {
        if(node->level > *max_height)
        {
            *max_height = node->level;
        }

	set_tree_height(node->left, max_height);

	set_tree_height(node->right, max_height);
    }
}

// ============== GPU
unsigned int get_array_size(unsigned int tree_height)
{
    unsigned int size_sum = 1;

    while(tree_height > 0)
    {
        size_sum += pow(2.0, (double)tree_height);

	tree_height--;
    }

    return size_sum;
}

void init_kd_tree_node_gpu(struct kd_tree_node_gpu* gpu_node, unsigned int dim)
{
    (*gpu_node).data = (double*)calloc(dim, sizeof(double));
    (*gpu_node).level = 0;
    (*gpu_node).metric = 0;
    (*gpu_node).dim = dim;
    (*gpu_node).left_child_index = 0;
    (*gpu_node).right_child_index = 0;
    (*gpu_node).parent_index = 0;

}

void convert_tree_to_array(
    struct kd_tree_node_cpu** cpu_node,
    struct kd_tree_node_gpu** gpu_node_array,
    unsigned int insert_index,
    unsigned int* max_size,
    unsigned int* index_array,
    unsigned int* index_array_insert
) {
    // check if current node is not null
    if((*cpu_node) != NULL)
    {
    	index_array[*index_array_insert] = insert_index;
    	*index_array_insert += 1;
        if(insert_index > *max_size)
        {
            *max_size = insert_index;
    	}
    	// allocate gpu node at current index
        //cudaMalloc((double**)&((*gpu_node_array)[insert_index].data), (*cpu_node)->dim * sizeof(double));
    	(*gpu_node_array)[insert_index].data = (double*)malloc(sizeof(double) * (*cpu_node)->dim);
        
    	//copy data from current cpu node to gpu node
        (*gpu_node_array)[insert_index].level = (*cpu_node)->level;
        (*gpu_node_array)[insert_index].metric = (*cpu_node)->metric;
        (*gpu_node_array)[insert_index].dim = (*cpu_node)->dim;
        for (int i = 0; i < (*cpu_node)->dim; i++)
    	{
            (*gpu_node_array)[insert_index].data[i] = (*cpu_node)->data[i];
    	}

        // cudaMemcpy((*gpu_node_array)[insert_index].data, (*cpu_node)->data, (*gpu_node_array)[insert_index].dim * sizeof(double), cudaMemcpyHostToDevice);
        
    	// initialize left and right indicies
	//if(((*cpu_node)->left) != NULL)
	//{
            (*gpu_node_array)[insert_index].left_child_index = (2 * insert_index) + 1;
	//}
	//else
	//{
            //(*gpu_node_array)[insert_index].left_child_index = -1;
	//}

	//if(((*cpu_node)->right) != NULL)
	//{
            (*gpu_node_array)[insert_index].right_child_index = (2 * insert_index) + 2;
	//}
	//else
	//{
            //(*gpu_node_array)[insert_index].right_child_index = -1;
	//}
        (*gpu_node_array)[insert_index].parent_index = (insert_index - 1) / 2;
        //printf("INSERT INDEX: %d; current level: %d\n", insert_index,
               //(*gpu_node_array)[insert_index].level);
    	//printf("INSERT INDEX: %d; current child indicies: %d, %d\n", insert_index,
                   //(*gpu_node_array)[insert_index].left_child_index, (*gpu_node_array)[insert_index].right_child_index);
    	// recurse left
    	convert_tree_to_array(&((*cpu_node)->left), gpu_node_array, (2 * insert_index) + 1, max_size, index_array, index_array_insert);

    	// recurse right
        convert_tree_to_array(&((*cpu_node)->right), gpu_node_array, (2 * insert_index) + 2, max_size, index_array, index_array_insert);
    }
}

void allocate_gpu_memory(
    struct kd_tree_node_cpu** cpu_nodes,
    struct kd_tree_node_gpu** gpu_nodes,
    int num_nodes
) {
  /*
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


    //cudaMalloc((void**)&(gpu_node.data), gpu_node.dim * sizeof(double));
    //cudaMemcpy(gpu_node.data, cpu_nodes[i]->data, gpu_node.dim * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy( &((*gpu_nodes)[i]) , &gpu_node, sizeof(struct kd_tree_node_gpu), cudaMemcpyHostToDevice);
  }*/
}
