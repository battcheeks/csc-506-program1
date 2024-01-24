/**
 * @file plot2.cu
 * @author Aniruddha Kulkarni (akulka24@ncsu.edu)
 * @brief
 *
 * Plot 2
 * C(x) = C(x) = 3*A(x)^4/D(x) + 2*B(x)^4 + 5*A(x)^2B(x)^2/(E(x)D(x)) +3*A(x)^2B(x) + 7*A(x)B(x)^2 + 9/D(x)^2
 *
 * To do:
 *
 * 1. Implement the base and optimized kernel
 * 2. Re-direct the std output to a txt file
 */
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#define OPT 1

__global__ void
plot2(const float *A, const float *B, float *C, float *D, float *E, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
#if OPT==1
    // Insert your optimized code below

    if (i < numElements)
    {
        float a_sq = A[i] * A[i];                    
        float b_sq = B[i] * B[i];                   
        // float d_inv_sq = 1.0f / (D[i] * D[i]);                
        // float e_inv = (1.0f / E[i]);       

        // C[i] = (3.0f * a_sq * a_sq * d_inv) +     
        //        (2.0f * b_sq * b_sq) +               
        //        (5.0f * a_sq * b_sq * e_inv_d_inv) + 
        //        (3.0f * a_sq * B[i]) +                  
        //        (7.0f * A[i] * b_sq) +                  
        //        (9.0f * d_inv * d_inv);

        // C[i] = a_sq * (3.0f * a_sq * (1.0f/D[i]) + 5.0f * b_sq * e_inv_d_inv + 3 * B[i]) + b_sq * (2.0f * b_sq + 7.0f * A[i]) + 9.0f * d_inv * d_inv;
        // C[i] = d_inv * (a_sq * (3.0f * a_sq + 5.0f * b_sq * e_inv_d_inv) + 9.0f*d_inv) + B[i] *(3.0f*a_sq + B[i] * (2 * b_sq + 7*A[i]));
        // C[i] = a_sq * (3.0f*B[i] + (1.0f/D[i])*(5.0f*b_sq*(1.0f/E[i]) + 3.0f*a_sq)) + b_sq*(2.0f*b_sq+7.0f*A[i]) + 9.0f*d_inv_sq;
        C[i] = (1.0f/D[i])*(a_sq*(3.0f*a_sq + 5.0f*b_sq*(1.0f/E[i])) + 9*(1.0f/D[i])) + B[i]*(3.0f*a_sq + B[i]*(2.0f*b_sq + 7.0f*A[i]));

        
    }

    // Insert your optimized code above
#endif


#if OPT==0
    if (i < numElements)
    {
        C[i] = (3*A[i]*A[i]*A[i]*A[i])/D[i] + 2*B[i]*B[i]*B[i]*B[i] + (5*A[i]*A[i]*B[i]*B[i])/(E[i]*D[i]) + 3*A[i]*A[i]*B[i] + 7*A[i]*B[i]*B[i] + 9/(D[i]*D[i]);
    }

#endif

}

/**
 * Host main routine
 * Read through the main function and understand the memory allocation, memory copy and freeing the memory. Then insert appropriate code for vector D.
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Plot 2 Vector Addition calculation for %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    float *h_D = (float *)malloc(size);
    float *h_E = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL || h_D == NULL || h_E == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
        h_D[i] = rand()/(float)RAND_MAX;
        h_E[i] = rand()/(float)RAND_MAX;
        // insert code here to allocate random variables to vector D
	
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
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

    float *d_E = NULL;
    err = cudaMalloc((void **)&d_E, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector E (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    
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

    err = cudaMemcpy(d_E, h_E, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector E from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    plot2<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C,d_D, d_E, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
   for (int i = 0; i < numElements; ++i)
    {

        if (fabs((h_C[i] - ((3 * h_A[i] * h_A[i] * h_A[i] * h_A[i]) / h_D[i] + 2 * h_B[i] * h_B[i] * h_B[i] * h_B[i] + (5 * h_A[i] * h_A[i] * h_B[i] * h_B[i]) / (h_E[i] * h_D[i]) + 3 * h_A[i] * h_A[i] * h_B[i] + 7 * h_A[i] * h_B[i] * h_B[i] + 9 / (h_D[i] * h_D[i])))/h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");
    // Free device global memory
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

    err = cudaFree(d_E);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector E (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_E);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

