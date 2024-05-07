#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>
#include <math.h>

#include "../include/kd_tree.cuh"


//Error checking GPU calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Mode 0 is CPU brute force - for checking tree results
// Mode 1 is CPU sequential implementation of kd-tree
// Mode 2 brute for query with GPU - for checking and comparing
// Mode 3 uses CPU to build the kd-tree and move to GPU, then uses GPU to query
// Mode 4 uses shared memory to move nodes in question to the tree
// Mode 5 uses 2D block
// Mode 6 uses 3D block
// ...
// #define MODE 0

//Define any constants here
#define BLOCKSIZE 128

using namespace std;


//function prototypes
void warmUpGPU();
void checkParams(unsigned int N, unsigned int DIM);

// cpu code
// brute force
void calcDistMatCPU(
    double* distanceMatrix,
    double* dataset,
    const unsigned int N,
    const unsigned int DIM
);
void queryDistMat(
    unsigned int* result,
    double* distanceMatrix,
    const double epsilon,
    const unsigned int N
);

// kd-tree
kd_tree_cpu* buildKdTreeCPU(const double* dataset, const unsigned int N, const unsigned int DIM);
void queryKdTreeCPU(
    struct kd_tree_cpu** tree,
    unsigned int* result,
    const double* dataset,
    const double epsilon,
    const unsigned int N,
    const unsigned int DIM
);

// gpu code
__global__ void init_node_data(struct kd_tree_node_gpu* gpu_nodes_array, double* data, int insert_index);

// brute force?
__global__ void calcDistMatGPU(
    double* distanceMatrix,
    double* dataset,
    const unsigned int N,
    const unsigned int DIM
);


__global__ void queryDistMatGPU(
    unsigned int* result,
    double* distanceMatrix,
    const double epsilon,
    const unsigned int N
);


// query kd-tree
__global__ void queryKdTreeGPU(
    unsigned int* result,
    struct kd_tree_node_gpu* node_array,
    unsigned int* indices,
    const double epsilon,
    const unsigned int N,
    const unsigned int DIM
);


__global__ void queryKdTreeGPUWithSharedMem(
    unsigned int* result,
    struct kd_tree_node_gpu* node_array,
    unsigned int* indices,
    const double epsilon,
    const unsigned int N,
    const unsigned int DIM
);



// handling data
void importDataset(
    char* fname,
    unsigned int N,
    unsigned int DIM,
    double* dataset
);



int main(int argc, char* argv[])
{
    printf("\nMODE: %d", MODE);
    warmUpGPU();

    char inputFname[500];
    unsigned int N = 0;
    unsigned int DIM = 0;
    double epsilon = 0;


    if (argc != 5)
    {
        fprintf(stderr, "Please provide the following on the command line: N (number of lines in the file), dimensionality (number of coordinates per point), epsilon, dataset filename.\n");
        exit(0);
    }

    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &DIM);
    sscanf(argv[3], "%f", &epsilon);
    strcpy(inputFname, argv[4]);

    checkParams(N, DIM);

    printf(
        "\nAllocating the following amount of memory for the dataset: %f GiB",
        (sizeof(double) * N * DIM) / (1024 * 1024 * 1024.0)
    );
    printf(
        "\nAllocating the following amount of memory for the distance matrix: %f GiB",
        (sizeof(double) * N * N) / (1024 * 1024 * 1024.0)
    );


    double tstartbuild = 0.0;
    double tendbuild = 0.0;
    double tstartquery = 0.0;
    double tendquery = 0.0;
    double* dataset = (double*)malloc(sizeof(double) * N * DIM);
    double* distanceMatrix = (double*)malloc(sizeof(double) * N * N);
    unsigned int* result = (unsigned int*)malloc(sizeof(unsigned int) * N);
    importDataset(inputFname, N, DIM, dataset);


    //CPU-only mode
    //It only computes the distance matrix but does not query the distance matrix
    if (MODE == 0)
    {
        double tstartcalc = omp_get_wtime();
        calcDistMatCPU(distanceMatrix, dataset, N, DIM);
        double tendcalc = omp_get_wtime();

        double tstartquery = omp_get_wtime();
        queryDistMat(result, distanceMatrix, epsilon, N);
        double tendquery = omp_get_wtime();

        unsigned int totalWithinEpsilon = 0;
        for (unsigned int i = 0; i < N; i += 1)
        {
            totalWithinEpsilon += result[i];
        }

        printf("\nTotal number of points within epsilon: %u\n", totalWithinEpsilon);
        printf("\nTime to calc the tree: %f", tendcalc - tstartcalc);
        printf("\nTime to query the tree: %f", tendquery - tstartquery);
        printf(
            "\n[MODE: %d, N: %d, E: %0.1f] Total time: %f\n",
            MODE, N, epsilon,
            (tendcalc - tstartcalc) + (tendquery - tstartquery)
        );

        return 0;
    }
    else if (MODE == 1)  // build and query kd-tree on CPU
    {
        tstartbuild = omp_get_wtime();
        struct kd_tree_cpu* tree = buildKdTreeCPU(dataset, N, DIM);
        tendbuild = omp_get_wtime();

        // printf("\n\nkd-tree:");
        // print_tree(tree);
        // printf("\n");

        tstartquery = omp_get_wtime();
        queryKdTreeCPU(&tree, result, dataset, epsilon, N, DIM);
        tendquery = omp_get_wtime();

        unsigned int totalWithinEpsilon = 0;
        for (unsigned int i = 0; i < N; i += 1)
        {
            totalWithinEpsilon += result[i];
        }

        printf("\nTotal number of points within epsilon: %u\n", totalWithinEpsilon);
        printf("\nTime to build the tree: %f", tendbuild - tstartbuild);
        printf("\nTime to query the tree: %f", tendquery - tstartquery);
        printf(
            "\n[MODE: %d, N: %d, E: %0.1f] Total time: %f\n",
            MODE, N, epsilon,
            (tendbuild - tstartbuild) + (tendquery - tstartquery)
        );

        return 0;
    }

    double tstart = omp_get_wtime();

    //Allocate memory for the dataset
    double* dev_dataset;
    gpuErrchk(cudaMalloc((double**)&dev_dataset, sizeof(double) * N * DIM));
    gpuErrchk(cudaMemcpy(dev_dataset, dataset, sizeof(double) * N * DIM, cudaMemcpyHostToDevice));

    //For part 1 that computes the distance matrix
    double* dev_distanceMatrix;
    gpuErrchk(cudaMalloc((double**)&dev_distanceMatrix, sizeof(double) * N * N));

    //For part 2 for querying the distance matrix
    unsigned int* resultSet = (unsigned int*)calloc(N, sizeof(unsigned int));
    unsigned int* dev_resultSet;
    gpuErrchk(cudaMalloc((unsigned int**)&dev_resultSet, sizeof(unsigned int) * N));
    gpuErrchk(cudaMemcpy(dev_resultSet, resultSet, sizeof(unsigned int) * N, cudaMemcpyHostToDevice));

    if (MODE == 2)  // brute force with GPU
    {
        unsigned int BLOCKDIM = BLOCKSIZE;
        unsigned int NBLOCKS = ceil(N*1.0 / BLOCKDIM*1.0);

        tstartbuild = omp_get_wtime();
        calcDistMatGPU<<<NBLOCKS, BLOCKDIM>>>(dev_distanceMatrix, dev_dataset, N, DIM);
        tendbuild = omp_get_wtime();

        tstartquery = omp_get_wtime();
        queryDistMatGPU<<<NBLOCKS, BLOCKDIM>>>(dev_resultSet, dev_distanceMatrix, epsilon, N);
        tendquery = omp_get_wtime();
    }
    else if (MODE == 3)  // build tree on CPU and query on GPU
    {
        unsigned int BLOCKDIM = BLOCKSIZE;
        unsigned int NBLOCKS = ceil(N*1.0 / BLOCKDIM*1.0);
        struct kd_tree_gpu* gpu_tree = NULL;

        tstartbuild = omp_get_wtime();
        struct kd_tree_cpu* tree = buildKdTreeCPU(dataset, N, DIM);
        tendbuild = omp_get_wtime();

        // move the tree onto the GPU - use another pair of time vars to measure the time it takes
        // to move the tree over to the GPU
        // initialize dev gpu node array
        struct kd_tree_node_gpu* dev_gpu_nodes_array;
       
        // initialize gpu node array on host
        // for now, this is just using the an arbritrary size that only works for data set of 100 points;
        // should figure out a way to calculate gpu node array size based of height of cpu tree;
        struct kd_tree_node_gpu* gpu_nodes_array = (struct kd_tree_node_gpu*)malloc(sizeof(struct kd_tree_node_gpu) * 16375);

        // this is redundant for now; instead of itereting through N, it should probably be through
        // calculated size of gpu node array based on height of the cpu tree
        for(unsigned int i = 0; i < N; i++)
        {
            init_kd_tree_node_gpu(&(gpu_nodes_array[i]), DIM);
        }
        
        // gpuErrchk(cudaMalloc((struct kd_tree_node_gpu**)&gpu_nodes_array, sizeof(struct kd_tree_node_gpu) * N));
        // convert kd tree into heap like structure
        unsigned int max_size = 0;
        unsigned int index_array[N];
        unsigned int index_array_insert = 0;
        convert_tree_to_array(&(tree->root), &gpu_nodes_array, 0, &max_size, index_array, &index_array_insert);
      	max_size += 1;
        
        double* dev_data_gpu;
        gpuErrchk(cudaMalloc((double**)&(dev_data_gpu), sizeof(double) * DIM));
        
        // copy over gpu node array from host to device
        // printf("\nleft child index at 16374: %d\n", gpu_nodes_array[16374].left_child_index);
        gpuErrchk(cudaMalloc((struct kd_tree_node_gpu**)&dev_gpu_nodes_array, sizeof(struct kd_tree_node_gpu) * max_size));
        gpuErrchk(cudaMemcpy(dev_gpu_nodes_array, gpu_nodes_array, sizeof(struct kd_tree_node_gpu) * max_size, cudaMemcpyHostToDevice));
        unsigned int insert_index = 0;
        for(unsigned int i = 0; i < N; i++)
        {
            insert_index = index_array[i];

            // copy data from current node in gpu node array to a device data array
            gpuErrchk(cudaMemcpy(dev_data_gpu, gpu_nodes_array[insert_index].data, sizeof(double) * DIM, cudaMemcpyHostToDevice));

            // initialize memory and data on dev gpu node array at the current index; has to be on device code :/
            init_node_data<<<1, 1>>>(dev_gpu_nodes_array, dev_data_gpu, insert_index);
        }
	
        tstartquery = omp_get_wtime();
        queryKdTreeGPU<<<NBLOCKS, BLOCKDIM>>>(dev_resultSet, dev_gpu_nodes_array, index_array, epsilon, N, DIM);
        tendquery = omp_get_wtime();
    }
    else if (MODE == 4)  // use shared memory
    {
        unsigned int BLOCKDIM = BLOCKSIZE;
        unsigned int NBLOCKS = ceil(N*1.0 / BLOCKDIM*1.0);
        struct kd_tree_gpu* gpu_tree = NULL;

        printf("\nMODE 4 IS NOT IMPLEMENTED YET!");
        /*
        tstartbuild = omp_get_wtime();
        struct kd_tree_cpu* tree = buildKdTreeCPU(dataset, N, DIM);
        tendbuild = omp_get_wtime();

        // move the tree onto the GPU

        tstartquery = omp_get_wtime();
        queryKdTreeGPUWithSharedMem<<<NBLOCKS, BLOCKDIM>>>(&gpu_tree, dev_resultSet, dev_dataset, epsilon, N, DIM);
        tendquery = omp_get_wtime();
        */
    }
    else if (MODE == 5)  // uses 2D block for querying
    {
        unsigned int BLOCKDIM = BLOCKSIZE;
        unsigned int NBLOCKS = ceil(N*1.0 / BLOCKDIM*1.0);
        struct kd_tree_gpu* gpu_tree = NULL;


        // build tree on CPU
        tstartbuild = omp_get_wtime();
        kd_tree_cpu* tree = buildKdTreeCPU(dataset, N, DIM);
        tendbuild = omp_get_wtime();

        printf("\nTime to build the tree: %f", tendbuild - tstartbuild);

        printf("\nMODE 5 IS NOT IMPLEMENTED YET!");
        /*
        tstartbuild = omp_get_wtime();
        struct kd_tree_cpu* tree = buildKdTreeCPU(dataset, N, DIM);
        tendbuild = omp_get_wtime();

        // move the tree onto the GPU

        tstartquery = omp_get_wtime();
        queryKdTreeGPUWithTwoDimBlock<<<NBLOCKS, BLOCKDIM>>>(&gpu_tree, dev_resultSet, dev_dataset, epsilon, N, DIM);
        tendquery = omp_get_wtime();
        */
    }
    else if (MODE == 6)  // uses 3D block for querying
    {
        unsigned int BLOCKDIM = BLOCKSIZE;
        unsigned int NBLOCKS = ceil(N*1.0 / BLOCKDIM*1.0);
        struct kd_tree_gpu* gpu_tree = NULL;

        printf("\nMODE 6 IS NOT IMPLEMENTED YET!");
        /*
        tstartbuild = omp_get_wtime();
        struct kd_tree_cpu* tree = buildKdTreeCPU(dataset, N, DIM);
        tendbuild = omp_get_wtime();

        // move the tree onto the GPU

        tstartquery = omp_get_wtime();
        queryKdTreeGPUWithThreeDimBlock<<<NBLOCKS, BLOCKDIM>>>(&gpu_tree, dev_resultSet, dev_dataset, epsilon, N, DIM);
        tendquery = omp_get_wtime();
        */
    }

    //Copy result set from the GPU
    gpuErrchk(cudaMemcpy(resultSet, dev_resultSet, sizeof(unsigned int) * N, cudaMemcpyDeviceToHost));

    //Compute the sum of the result set array
    unsigned int totalWithinEpsilon = 0;

    //Write code here
    for(int resultIndex = 0; resultIndex < N; resultIndex += 1)
    {
        totalWithinEpsilon += resultSet[resultIndex];
    }

    printf("\nTotal number of points within epsilon: %u", totalWithinEpsilon);

    double tend = omp_get_wtime();

    printf("\nTime to build the tree: %f\n", tendbuild - tstartbuild);
    printf("\nTime to query the tree: %f\n", tendquery - tstartquery);

    printf(
        "\n[MODE: %d, N: %d, E: %0.1f] Total time: %f\n",
        MODE, N, epsilon,
        tend - tstart
    );

    printf("\n\n");
    return 0;
}



void warmUpGPU()
{
    printf("\nWarming up GPU for time trialing...\n");
    cudaDeviceSynchronize();
    return;
}


void checkParams(unsigned int N, unsigned int DIM)
{
    if (N <= 0 || DIM <= 0)
    {
        fprintf(stderr, "\n Invalid parameters: Error, N: %u, DIM: %u", N, DIM);
        fprintf(stderr, "\nReturning");
        exit(0);
    }
}


void importDataset(
        char* fname,
        unsigned int N,
        unsigned int DIM,
        double* dataset
) {
    FILE *fp = fopen(fname, "r");

    if (!fp)
    {
        fprintf(stderr, "Unable to open file\n");
        fprintf(stderr, "Error: dataset was not imported. Returning.");
        exit(0);
    }

    unsigned int bufferSize = DIM * 10;

    char buf[bufferSize];
    unsigned int rowCnt = 0;
    unsigned int colCnt = 0;
    while (fgets(buf, bufferSize, fp) && rowCnt < N)
    {
        colCnt = 0;

        char *field = strtok(buf, ",");
        double tmp;
        sscanf(field, "%lf", &tmp);

        dataset[rowCnt * DIM + colCnt] = tmp;


        while (field)
        {
            colCnt += 1;
            field = strtok(NULL, ",");

            if (field!=NULL)
            {
                double tmp;
                sscanf(field,"%lf",&tmp);
                dataset[rowCnt*DIM+colCnt]=tmp;
            }
        }

        rowCnt += 1;
    }

    fclose(fp);
}



// cpu
// brute force
void calcDistMatCPU(double* distanceMatrix, const double* dataset, const unsigned int N, const unsigned int DIM)
{
    for (unsigned int i = 0; i < N; i += 1)
    {
        for (unsigned int j = 0; j < N; j += 1)
        {
            double dist = 0.0;

            for (unsigned int d = 0; d < DIM; d += 1)
            {
                dist += (dataset[i * DIM + d] - dataset[j * DIM + d])
                    * (dataset[i * DIM + d] - dataset[j * DIM + d]);
            }

            distanceMatrix[i * N + j] = sqrt(dist);
        }
    }
}


void queryDistMat(unsigned int* result, const double* distanceMatrix, const double epsilon, const unsigned int N)
{
    for (unsigned int i = 0; i < N; i += 1)
    {
        for (unsigned int j = 0; j < N; j += 1)
        {
            if (distanceMatrix[i * N + j] <= epsilon)
            {
                result[i] += 1;
            }
        }
    }
}


// kd-tree
kd_tree_cpu* buildKdTreeCPU(const double* dataset, const unsigned int N, const unsigned int DIM)
{
    kd_tree_cpu* tree;

    init_kd_tree_cpu(&tree);

    for (unsigned int p = 0; p < N; p += 1)
    {
        kd_tree_node_cpu* node;
        double data[DIM];

        for (unsigned int d = 0; d < DIM; d += 1)
        {
            data[d] = dataset[p * DIM + d];
        }

        init_kd_tree_node_cpu(&node, data, DIM);
        insert(&tree, &node);
    }

    return tree;
}


void queryKdTreeCPU(kd_tree_cpu** tree, unsigned int* result, const double* dataset, const double epsilon, const unsigned int N, const unsigned int DIM)
{
    double query[2];
    unsigned int count;
    for (unsigned int p = 0; p < N; p += 1)
    {
        count = 0;

        for (unsigned int d = 0; d < DIM; d += 1)
        {
            query[d] = dataset[p * DIM + d];
        }

        points_within_epsilon_cpu(tree, query, epsilon, &count);

        result[p] = count;
    }
}



//===================================================//
//                       GPU                         //
//===================================================//

//querying tree/heap
__global__ void queryKdTreeGPU(struct kd_tree_node_gpu* gpu_nodes_array, double* testData)
{
    //testing/debugging for now
    //gpu_nodes_array[0].level = 69;
    //gpu_nodes_array[0].data[0] = 420;
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;

    if(tid == 0)
    {
	//gpu_nodes_array[0].data = (double*)malloc(sizeof(double) * gpu_nodes_array[0].dim);
        //gpu_nodes_array[0].data[0] = 42069;
	testData[0] = gpu_nodes_array[16374].data[0];
    }

    return;
}


//initialize data pointers on device
__global__ void init_node_data(struct kd_tree_node_gpu* gpu_nodes_array, double* data, int insert_index)
{
    gpu_nodes_array[insert_index].data = (double*)malloc(sizeof(double) * gpu_nodes_array[insert_index].dim);
    for(int i = 0; i < gpu_nodes_array[insert_index].dim; i++)
    {
        gpu_nodes_array[insert_index].data[i] = data[i];
    }
}


// brute force?
__global__ void calcDistMatGPU(
    double* distanceMatrix,
    double* dataset,
    const unsigned int N,
    const unsigned int DIM
) {
    const unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    double dist;


    if (tid >= N)
    {
        return;
    }

    for (unsigned int p = 0; p < N; p += 1)
    {
        dist = 0.0;

        for (unsigned int d = 0; d < DIM; d += 1)
        {
            dist += (dataset[tid * DIM + d] - dataset[p * DIM + d])
                * (dataset[tid * DIM + d] - dataset[p * DIM + d]);
        }

        distanceMatrix[tid * N + p] = sqrt(dist);
    }


    return;
}


__global__ void queryDistMatGPU(
    unsigned int* result,
    double* distanceMatrix,
    const double epsilon,
    const unsigned int N
) {
    const unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    unsigned int neighbors = 0;

    if (tid >= N)
    {
        return;
    }

    for (unsigned int p = 0; p < N; p += 1)
    {
        if (distanceMatrix[tid * N + p] <= epsilon)
        {
            neighbors += 1;
        }
    }

    result[tid] = neighbors += 1;


    return;
}


// query kd-tree
__global__ void queryKdTreeGPU(
        unsigned int* result,
        struct kd_tree_node_gpu* node_array,
        unsigned int* indices,
        const double epsilon,
        const unsigned int N,
        const unsigned int DIM
) {
    const unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    double dist = 0.0;
    double dist_prime = 0.0;
    double* query = node_array[indices[tid]].data;
    struct kd_tree_node_gpu* working = &node_array[0];
    struct kd_tree_node_gpu* first = NULL;
    struct kd_tree_node_gpu* second = NULL;

    
    if (tid > N)
    {
        return;
    }
    // allcate space to store seconds which must be visited

    // secondary index for entering new points in 'seconds' called 's'
    // loop over seconds array
    {
        // label: visit_subtree

        // loop until end of tree
        {
            // 1. calc dist

            // 2. check point within 'epsilon'
            {
                // 2a. update result
            }

            // 3. check query less than metric
            {
                // 3a. set first to left
                // 3b. set second to right
            }
            // 4. otherwise, assume query greater than metric
            {
                // 4a. set first to right
                // 4b. set second to left
            }

            // 5. calc dist to split axis

            // 6. check second exists and check split axis within 'epsilon'
            {
                // 6a. save second at 's'
                // 6b. update 's'
                // 6c. set second at 's' to 0
            }

            // 7. set working to first
        }

        // 8. check need to visit a second
        {
            // 8a. set working to second at 'i' in seconds
            // 8b. update 'i'
            // 8c. go to label 'visit_subtree'
        }
        // 9. otherwise, assume query is finished
        {
            // 9a. break loop
        }
    }
    
    
    return;
}


__global__ void queryKdTreeGPUWithSharedMem(
    struct kd_tree_node_gpu** node_array,
    unsigned int* result,
    const double epsilon,
    const unsigned int N,
    const unsigned int DIM
) {
    return;
}
