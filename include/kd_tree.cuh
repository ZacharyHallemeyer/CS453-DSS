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
    float metric;
    // dimensions of the point
    unsigned int dim;
    // data this node represents
    float* data;
    struct kd_tree_node_cpu* left;
    struct kd_tree_node_cpu* right;
    struct kd_tree_node_cpu* parent;
};


// function prototypes
void __points_within_epsilon(
    struct kd_tree_node_cpu** node,
    const float* query,
    const float epsilon,
    unsigned int* count
);


void __insert(
    struct kd_tree_node_cpu** parent,
    struct kd_tree_node_cpu** node,
    struct kd_tree_node_cpu** new_node,
    const unsigned int level
);


void __free_kd_tree_cpu(struct kd_tree_node_cpu** node);


void points_within_epsilon(
    struct kd_tree_cpu** tree,
    const float* query,
    const float epsilon,
    unsigned int* count
);


void free_kd_tree_cpu(struct kd_tree_cpu** tree);


void init_kd_tree_cpu(struct kd_tree_cpu** tree);


void init_kd_tree_node_cpu(
    struct kd_tree_node_cpu** node,
    const float* data,
    const unsigned int dim,
    const unsigned int level
);


void insert(struct kd_tree_cpu** tree, struct kd_tree_node_cpu** new_node);


void print_tree(struct kd_tree_cpu* tree);


void __print_tree(struct kd_tree_node_cpu* node);


// ============ GPU

struct kd_tree_node_gpu {
  unsigned int level;
  float metric;
  unsigned int dim;
  float* data;
  int left_child_index;
  int right_child_index;
  int parent_index;
};


void allocate_tree_gpu(struct kd_tree_node_cpu** cpu_node, struct kd_tree_node_gpu** gpu_node_array, int insert_index);
void allocate_gpu_memory(struct kd_tree_node_cpu** cpu_nodes, struct kd_tree_node_gpu** gpu_nodes, int num_nodes);
