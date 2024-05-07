#include <math.h>
#include <stdlib.h>


// structures
struct kd_tree_cpu
{
    // provides an easy way to get the height of the tree
    unsigned int size;
    unsigned int height;
    struct kd_tree_node_cpu* root;
};

struct kd_tree_node_cpu
{
    // provides an easy way for nodes to compare dimensions
    unsigned int level;
    // the dimension along which to split the space
    double metric;
    // dimensions of the point
    unsigned int dim;
    // data this node represents
    double* data;
    struct kd_tree_node_cpu* left;
    struct kd_tree_node_cpu* right;
    struct kd_tree_node_cpu* parent;
};


// function prototypes
void __points_within_epsilon_cpu(
    struct kd_tree_node_cpu** node,
    double* query,
    const double epsilon,
    unsigned int* count
);


void __insert(
    struct kd_tree_node_cpu** parent,
    struct kd_tree_node_cpu** node,
    struct kd_tree_node_cpu** new_node,
    const unsigned int level
);


void __free_kd_tree_cpu(struct kd_tree_node_cpu** node);


void points_within_epsilon_cpu(
    struct kd_tree_cpu** tree,
    double* query,
    const double epsilon,
    unsigned int* count
);


void free_kd_tree_cpu(struct kd_tree_cpu** tree);


void init_kd_tree_cpu(struct kd_tree_cpu** tree);


void init_kd_tree_node_cpu(
    struct kd_tree_node_cpu** node,
    const double* data,
    const unsigned int dim
);


void insert(struct kd_tree_cpu** tree, struct kd_tree_node_cpu** new_node);


void print_tree(struct kd_tree_cpu* tree);


void __print_tree(struct kd_tree_node_cpu* node);



// ============ GPU
struct kd_tree_tree
{
    unsigned int size;
    unsigned int height;
    struct kd_tree_node_gpu* root;
};

struct kd_tree_node_gpu
{
    unsigned int level;
    double metric;
    unsigned int dim;
    double* data;
    unsigned int visited;
    unsigned int left_child_index;
    unsigned int right_child_index;
    unsigned int parent_index;
};


void init_kd_tree_node_gpu(struct kd_tree_node_gpu* gpu_node, int dim);


void convert_tree_to_array(
    struct kd_tree_node_cpu** cpu_node,
    struct kd_tree_node_gpu** gpu_node_array,
    unsigned int insert_index,
    unsigned int* max_size,
    unsigned int* index_array,
    unsigned int* index_array_insert
);


void __points_within_epsilon_gpu(
    struct kd_tree_node_gpu** tree,
    double* query,
    const double epsilon,
    unsigned int* count
);


void allocate_gpu_memory(
    struct kd_tree_node_cpu** cpu_nodes,
    struct kd_tree_node_gpu** gpu_nodes,
    int num_nodes
);


void points_within_epsilon_gpu(
    struct kd_tree_gpu** tree,
    double* query,
    const double epsilon,
    unsigned int* count
);
