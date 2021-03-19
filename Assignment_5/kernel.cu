
#include "common.h"
#include "timer.h"

#define BLOCK_DIM 1024
#define SUCCESS 0

__global__ void reduce_kernel(float* input, float* partialSums, unsigned int N) {

    unsigned int segment = 2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    __shared__ float input_s[BLOCK_DIM];

    input_s[threadIdx.x] = (i < N) * input[i] + (i + BLOCK_DIM < N) * input[i+BLOCK_DIM];
    
    __syncthreads();

    for(unsigned int stride=BLOCK_DIM/2; stride > 0; stride /= 2) {

        if (threadIdx.x > BLOCK_DIM - stride - 1) {
            input_s[threadIdx.x] += input_s[ threadIdx.x - stride ];
        }

        __syncthreads();

    }

    if (threadIdx.x == blockDim.x - 1) {
        partialSums[blockIdx.x] = input_s[threadIdx.x];
    }


}

float reduce_gpu(float* input, unsigned int N) {

    int cudaError = SUCCESS;

    Timer timer;

  // Allocate memory
    startTime(&timer);
    float *input_d;
    cudaError = cudaMalloc((void**) &input_d, N*sizeof(float));
    printf("\033[0;31m");
    if(cudaError != SUCCESS) printf("There was a problem allocating memory to input_d\n");
    printf("\033[0m");
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

  // Copy data to GPU
    startTime(&timer);
    cudaError = cudaMemcpy(input_d, input, N*sizeof(float), cudaMemcpyHostToDevice);
    printf("\033[0;31m");
    if(cudaError != SUCCESS) printf("There was a problem copying memory to input_d\n");
    printf("\033[0m");
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

  // Allocate partial sums
    startTime(&timer);
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2*numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;
    float* partialSums = (float*) malloc(numBlocks*sizeof(float));
    float *partialSums_d;
    cudaError = cudaMalloc((void**) &partialSums_d, numBlocks*sizeof(float));
    printf("\033[0;31m");
    if(cudaError != SUCCESS) printf("There was a problem allocating memory for partialSums_d\n");
    printf("\033[0m");
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Partial sums allocation time");

  // Call kernel
    startTime(&timer);
    reduce_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, partialSums_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

  // Copy data from GPU
    startTime(&timer);
    cudaError = cudaMemcpy(partialSums, partialSums_d, numBlocks*sizeof(float), cudaMemcpyDeviceToHost);
    printf("\033[0;31m");
    if(cudaError != SUCCESS) printf("There was a problem copying memory from partialSums_d\n");
    printf("\033[0m");
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

  // Reduce partial sums on CPU
    startTime(&timer);
    float sum = 0.0f;
    for(unsigned int i = 0; i < numBlocks; ++i) {
        sum += partialSums[i];
    }
    stopTime(&timer);
    printElapsedTime(timer, "Reduce partial sums on host time");

  // Free memory
    startTime(&timer);
    cudaError = cudaFree(input_d);
    printf("\033[0;31m");
    if(cudaError != SUCCESS) printf("There was a problem deallocating memory for input_d\n");
    printf("\033[0m");
    free(partialSums);
    cudaError = cudaFree(partialSums_d);
    printf("\033[0;31m");
    if(cudaError != SUCCESS) printf("There was a problem deallocating memory for partialSums_d\n");
    printf("\033[0m");
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

    return sum;

}

