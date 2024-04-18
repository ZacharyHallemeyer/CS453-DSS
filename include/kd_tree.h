#include <stdlib.h>


// structures
struct kd_tree
{
    // provides an easy way to get the height of the tree
    unsigned int height;
    struct kd_tree_node* head;
};

struct kd_tree_node
{
    // provides an easy way for nodes to compare dimensions
    unsigned int level;
    // the dimension along which to split the space
    float metric;
    // dimensions of the point
    unsigned int dim;
    // data this node represents
    float* data;
    struct kd_tree_node* left;
    struct kd_tree_node* right;
};


// function prototypes
void init_kd_tree(struct kd_tree** tree);


void init_kd_tree_node(
    struct kd_tree_node** node,
    const float* data,
    const unsigned int dim,
    const unsigned int level
);


void insert(struct kd_tree_node** node, struct kd_tree_node** new_node);


void print_tree(struct kd_tree_node* node);


void free_kd_tree(struct kd_tree_node** node);
