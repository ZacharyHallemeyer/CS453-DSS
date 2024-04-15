# Distance Similarity Searches (kd-tree and heap-like data structure)


## Works Cited
### kd tree: https://en.wikipedia.org/wiki/K-d_tree
* a kd tree is a space partitioning data structure where every sub-space is
  split into two half-spaces
* each sub-tree contains points on either side of the sub-space

Consider the following pseudocode:
```
struct Point
{
    label: String,
    x: float,
    y: float
}


let a = Point { label: "A", x:  0.0, y: 0.0 }
let b = Point { label: "B", x: -1.0, y: 0.0 }
let c = Point { label: "C", x:  1.0, y: 0.0 }
```
If we were to construct a tree to store these points, with point `b` as the
root, our tree would look something like this:
```
  b
 / \
a   c
```
The tree is initially split using the `label` field of the `Point` structure.
This causes `b` to be the root, `a` to be to the left of `b`, and `c` to be to
the right of `b` in the tree.

What if we added another node? We'll have to choose a new dimension along which
to split: How about the `x` field?
```
let k = Point { label: "K", x: -2.0, y: 2.0 }
```
The tree now looks like this:
```
    b
   / \
  a   c
 /
k
```
This appears counter intuative because 'k' has a greater value than 'a'
alphabetically; however, we chose to split this level in the kd-tree along the
`x` field of the `Point` structure. Because the point `k` has an x position
smaller than `b` and `a`, `k` goes to the left of `b` and to the left of `a` in
the kd-tree. This continues for all points until there are no more points in to
add.

Additionally, choosing how to sort nodes is, effectively a modulo: if you run
out of "dimensions" with which to sort, you just go back to the first
dimension.


### Real-Time KD-Tree Construction on Graphics Hardware: http://www.kunzhou.net/2008/kdtree.pdf


### Massively parallel KD-tree construction and nearest neighbor search algorithms: https://arizona-nau.primo.exlibrisgroup.com/discovery/fulldisplay?docid=cdi_ieee_primary_7169256&context=PC&vid=01NAU_INST:01NAU&lang=en&search_scope=MyInst_and_CI&adaptor=Primo%20Central&tab=Everything&query=any,contains,Massively%20parallel%20KD-tree%20construction%20and%20nearest%20neighbor%20search%20algorithms&facet=rtype,exclude,reviews


### An Improved Method to Build the KD Tree Based on Presorted Results: https://arizona-nau.primo.exlibrisgroup.com/discovery/fulldisplay?docid=cdi_ieee_primary_9237636&context=PC&vid=01NAU_INST:01NAU&lang=en&search_scope=MyInst_and_CI&adaptor=Primo%20Central&tab=Everything&query=any,contains,balanced%20kd-tree%20construction&offset=0
The main idea of this article is to tackle the problem of balance within a
kd-tree. "Balance" refers to when one side of the tree contains more sub-trees
than the other. For example, the worst case scenario for a binary search tree
(BST) is if the tree becomes a list:
```
a
 \
  b
   \
    c
     \
      d
       ...
```
The solution presented by others has been to find the "median point" along
which to split the space which would result in each space to be split, more or
less, down the middle. Unfortunately, most of the solutions presented at the
time of the above paper's publication had results uncertain results or the
methods were time consuming. The above publication seeks to provide a solution.
