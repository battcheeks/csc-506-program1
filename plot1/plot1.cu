/**
 * @file Plot1.cu
 * @author Aniruddha Kulkarni (akulka24@ncsu.edu)
 * @brief
 *
 * Plot 1
 * C(x) = 4A(x)4– 2*A(x)2D(x)+ 8*A(x)3B(x) + 7*A(x)2B(x)2 + 5*A(x)B(x)3 + 2B(x)2 + 3*B(x)4 + 1
 *
 * To do:
 *
 * 1. Implement the base and optimized kernel
 * 2. Re-direct the std output to a txt file
 */

#include <stdio.h>
#include <cuda_runtime.h>

/**
 * @brief
 *
 */
#define OPT 1

/**
 * @brief
 *
 * @param A
 * @param B
 * @param C
 * @param D
 * @param numElements
 * @return __global__
 */
__global__ void plot1(const float *A, const float *B, float *C, float *D, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

#if OPT == 1

    if (i < numElements)
    {
        float asq = A[i] * A[i];
        float bsq = B[i] * B[i];

        C[i] = asq * (4 * asq - 2 * D[i] + 8 * A[i] * B[i] + 7 * bsq) + bsq * (5 * A[i] * B[i] + 2 + 3 * bsq) + 1;
    }
#endif

#if OPT == 0

    if (i < numElements)
    {
        C[i] = 4 * A[i] * A[i] * A[i] * A[i] - 2 * A[i] * A[i] * D[i] + 8 * A[i] * A[i] * A[i] * B[i] + 7 * A[i] * A[i] * B[i] * B[i] + 5 * A[i] * B[i] * B[i] * B[i] + 2 * B[i] * B[i] + 3 * B[i] * B[i] * B[i] * B[i] + 1;
    }

#endif
}

int main(void)
{
    cudaError_t err = cudaSuccess;

    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Plot 1 Vector Addition calculation for %d elements]\n", numElements);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_D = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL || h_D == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_D[i] = rand() / (float)RAND_MAX;
    }

    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_D = NULL;
    err = cudaMalloc((void **)&d_D, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector D (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_D, h_D, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector D from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    plot1<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i)
    {

        if (fabs((h_C[i] - (4 * h_A[i] * h_A[i] * h_A[i] * h_A[i] - 2 * h_A[i] * h_A[i] * h_D[i] + 8 * h_A[i] * h_A[i] * h_A[i] * h_B[i] + 7 * h_A[i] * h_A[i] * h_B[i] * h_B[i] + 5 * h_A[i] * h_B[i] * h_B[i] * h_B[i] + 2 * h_B[i] * h_B[i] + 3 * h_B[i] * h_B[i] * h_B[i] * h_B[i] + 1)) / h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    err = cudaFree(d_A);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_D);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector D (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}
