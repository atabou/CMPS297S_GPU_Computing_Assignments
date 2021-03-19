
#include "common.h"
#include "timer.h"


// GRID PARAMETERS

#define NUM_THREADS_X_DIRECTION 32
#define NUM_THREADS_Y_DIRECTION 32


// CUDA ERROR CODES

#define SUCCESS 0

__global__ void mm_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    // TODO


    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;

    if ( row < M && col < N ) {

        float dot_product = 0.0f;

        for( unsigned int i=0; i < K; i++ ) {
            dot_product += A[ row*K + i ] * B[ i*N + col ];
        }
        
        C[ row*N + col ] = dot_product;
    
    }
}

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    Timer timer;

    int cudaErrorCode = SUCCESS;

    // Allocate GPU memory
    startTime(&timer);

    // TODO

    float *A_d;
    float *B_d;
    float *C_d;

    printf("%d, %d, %d \n", M, N, K);

    cudaErrorCode = cudaMalloc((void **) &A_d, sizeof(float)*M*K);

    if( cudaErrorCode != SUCCESS ) {
        printf("There was an error allocating GPU memory for matrix A. Error Code: %d", cudaErrorCode);
    }

    cudaErrorCode = cudaMalloc((void **) &B_d, sizeof(float)*K*N);

    if( cudaErrorCode != SUCCESS ) {
        printf("There was an error allocating GPU memory for matrix B. Error Code: %d", cudaErrorCode);
    }

    cudaErrorCode = cudaMalloc((void **) &C_d, sizeof(float)*M*N);

    if( cudaErrorCode != SUCCESS ) {
        printf("There was an error allocating GPU memory for matrix C. Error Code: %d", cudaErrorCode);
    }

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);

    // TODO

    cudaErrorCode = cudaMemcpy( A_d, A, M*K*sizeof(float), cudaMemcpyHostToDevice );

    if ( cudaErrorCode != SUCCESS ) {
        printf("There was a problem copying A's data to A_d. Error Code: %d", cudaErrorCode);
    }

    cudaErrorCode = cudaMemcpy( B_d, B, K*N*sizeof(float), cudaMemcpyHostToDevice );

    if ( cudaErrorCode != SUCCESS ) {
        printf("There was a problem copying B's data to B_d. Error Code: %d", cudaErrorCode);
    }

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);

    // TODO
    dim3 numThreadsPerBlock(NUM_THREADS_X_DIRECTION, NUM_THREADS_Y_DIRECTION);
    dim3 numBlocks(     ( N + NUM_THREADS_X_DIRECTION - 1 ) / NUM_THREADS_X_DIRECTION,  ( M + NUM_THREADS_Y_DIRECTION - 1 ) / NUM_THREADS_Y_DIRECTION      );

    mm_kernel <<< numBlocks, numThreadsPerBlock >>>( A_d, B_d, C_d, M, N, K ); 

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);

    // TODO
 
    cudaMemcpy( C, C_d, M*N*sizeof(float), cudaMemcpyDeviceToHost );

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    // TODO

    cudaErrorCode = cudaFree(A_d);

    if(cudaErrorCode != SUCCESS ) {
        printf("There was an error deallocating A_d's memory on the GPU. Error Code: %d", cudaErrorCode);
    }

    cudaErrorCode = cudaFree(B_d);
    
    if(cudaErrorCode != SUCCESS ) {
        printf("There was an error deallocating B_d's memory on the GPU. Error Code: %d", cudaErrorCode);
    }
    
    cudaErrorCode = cudaFree(C_d);
    
    if(cudaErrorCode != SUCCESS ) {
        printf("There was an error deallocating C_d's memory on the GPU. Error Code: %d", cudaErrorCode);
    }

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

