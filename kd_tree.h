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
    // dimensions of the point represented by this node
    float* point;
    struct kd_tree_node* next;
};


// function prototypes
void init_kd_tree(
    struct kd_tree* tree,
    const float* data,
    const unsigned int DIM
);

void init_kd_tree_node(
    struct kd_tree_node* node,
    const float* data,
    const unsigned int DIM,
    const unsigned int level
);

void insert_data(struct kd_tree* tree, const float* data);
