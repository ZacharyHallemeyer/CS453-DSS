#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <complex.h>
#include <math.h>

#include "kd_tree.h"


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
// Mode 2 uses CPU to build the kd-tree and GPU to query
// Mode 3 ...
// ...
#define MODE 0

//Define any constants here
#define BLOCKSIZE 128

using namespace std;


//function prototypes
void warmUpGPU();
void checkParams(unsigned int N, unsigned int DIM);

// cpu code
// brute force
void calcDistMatCPU(float* distanceMatrix, const float* dataset, const unsigned int N, const unsigned int DIM);
void calcQueryDistMat(unsigned int* result, const float* distanceMatrix, const float epsilon, const unsigned int N, const unsigned int DIM);

// gpu code

// handling data
void importDataset(
    char* fname,
    unsigned int N,
    unsigned int DIM,
    float* dataset
);



int main(int argc, char* argv[])
{
    printf("\nMODE: %d", MODE);
    warmUpGPU();

    char inputFname[500];
    unsigned int N = 0;
    unsigned int DIM = 0;
    float epsilon = 0;


    if (argc != 5)
    {
        fprintf(stderr, "Please provide the following on the command line: N (number of lines in the file), dimensionality (number of coordinates per point), epsilon, dataset filename.\n");
        exit(0);
    }

    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &DIM);
    sscanf(argv[3], "%f", &epsilon);
    strcpy(inputFname,argv[4]);

    checkParams(N, DIM);

    printf(
        "\nAllocating the following amount of memory for the dataset: %f GiB",
        (sizeof(float) * N * DIM) / (1024 * 1024 * 1024.0)
    );
    printf(
        "\nAllocating the following amount of memory for the distance matrix: %f GiB",
        (sizeof(float) * N * N) / (1024 * 1024 * 1024.0)
    );


    float* dataset = (float*)malloc(sizeof(float) * N * DIM);
    float* distanceMatrix = (float*)malloc(sizeof(float) * N * N);
    unsigned int* result = (unsigned int*)malloc(sizeof(unsigned int) * N);
    importDataset(inputFname, N, DIM, dataset);


    //CPU-only mode
    //It only computes the distance matrix but does not query the distance matrix
    if (MODE == 0)
    {
        double tstart = omp_get_wtime();

        calcDistMatCPU(distanceMatrix, dataset, N, DIM);
        calcQueryDistMat(result, distanceMatrix, N, DIM);

        unsigned int totalWithinEpsilon = 0;
        for (unsigned int i = 0; i < N; i += 1)
        {
            totalWithinEpsilon += result[i];
        }

        double tend = omp_get_wtime();

        printf("\nTotal number of points within epsilon: %u", totalWithinEpsilon);
        printf("\n[MODE: %d, N: %d] Total time: %f", MODE, N, tend - tstart);

        return 0;
    }

    double tstart=omp_get_wtime();

    //Allocate memory for the dataset
    float* dev_dataset;
    gpuErrchk(cudaMalloc((float**)&dev_dataset, sizeof(float) * N * DIM));
    gpuErrchk(cudaMemcpy(dev_dataset, dataset, sizeof(float) * N * DIM, cudaMemcpyHostToDevice));

    //For part 1 that computes the distance matrix
    float* dev_distanceMatrix;
    gpuErrchk(cudaMalloc((float**)&dev_distanceMatrix, sizeof(float) * N * N));

    //For part 2 for querying the distance matrix
    unsigned int* resultSet = (unsigned int*)calloc(N, sizeof(unsigned int));
    unsigned int* dev_resultSet;
    gpuErrchk(cudaMalloc((unsigned int**)&dev_resultSet, sizeof(unsigned int) * N));
    gpuErrchk(cudaMemcpy(dev_resultSet, resultSet, sizeof(unsigned int) * N, cudaMemcpyHostToDevice));

    //Baseline kernels
    if (MODE == 1)
    {
        unsigned int BLOCKDIM = BLOCKSIZE;
        unsigned int NBLOCKS = ceil(N*1.0/BLOCKDIM);

        // Call baseline kernel here
        // TODO: IMPLEMENT GPU IMPLEMENTATION
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

    printf("\n[MODE: %d, N: %d] Total time: %f", MODE, N, tend-tstart);

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
        float* dataset
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
void calcDistMatCPU(float* distanceMatrix, const float* dataset, const unsigned int N, const unsigned int DIM)
{
    for (unsigned int i = 0; i < N; i += 1)
    {
        for (unsigned int j = 0; j < N; j += 1)
        {
            float dist = 0.0;

            for (unsigned int d = 0; d < DIM; d += 1)
            {
                dist += (dataset[i * DIM] + d] - dataset[i * DIM + c + 1])
                    * (dataset[i * DIM + d] - dataset[i * DIM + c + 1]);
            }

            distanceMatrix[i * N + j] = sqrt(dist);
        }
    }
}


void calcQueryDistMat(unsigned int* result, const float* distanceMatrix, const float epsilon, const unsigned int N, const unsigned int DIM)
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
