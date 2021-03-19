
#include "common.h"
#include "timer.h"

#define TILE_DIM 32
#define SUCCESS 0

__global__ void mm_tiled_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    // TODO

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];
    
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for( unsigned int tile = 0; tile < (float) K / TILE_DIM; tile++ ) {

        if( tile == ( K / TILE_DIM ) && threadIdx.x >= K % TILE_DIM ) {    
            A_s[threadIdx.y][threadIdx.x] = 0.0f;
        } else {
            A_s[threadIdx.y][threadIdx.x] = A[ row*K + tile*TILE_DIM + threadIdx.x ];
        }

        if( tile == ( K / TILE_DIM ) && threadIdx.y >=  K % TILE_DIM ) {    
            B_s[threadIdx.y][threadIdx.x] = 0.0f;
        } else {
            B_s[threadIdx.y][threadIdx.x] = B[ tile*TILE_DIM*N + threadIdx.y*N + col ];
        }

        __syncthreads();
  
        for( unsigned int i=0; i<TILE_DIM; i++ ) {
            sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
        }
 
        __syncthreads();
  
    }

    if( row < M && col < N ) {
        C[row*N + col] = sum;
    }

}

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {





    Timer timer;

    int cudaErrorCode = SUCCESS;

    // Allocate GPU memory
    startTime(&timer);



    // TODO

// -------------------------------------------START------------------------------------------------------

    float* A_d;
    float* B_d;
    float* C_d;

    cudaErrorCode = cudaMalloc( (void **) &A_d, M*K*sizeof(float) );
    
    if( cudaErrorCode != SUCCESS ) {
        printf("There was an error allocating memory for A_d. Error Code: %d\n", cudaErrorCode);
    }

    cudaErrorCode = cudaMalloc( (void **) &B_d, K*N*sizeof(float) );

    if( cudaErrorCode != SUCCESS ) {
        printf("There was an error allocating memory for A_d. Error Code: %d\n", cudaErrorCode);
    }
    
    cudaErrorCode = cudaMalloc( (void **) &C_d, M*N*sizeof(float) );
    
    if( cudaErrorCode != SUCCESS ) {
        printf("There was an error allocating memory for A_d. Error Code: %d\n", cudaErrorCode);
    }

// --------------------------------------------END-------------------------------------------------------





    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);



    // TODO

// -------------------------------------------START------------------------------------------------------

    cudaErrorCode = cudaMemcpy( A_d, A, M*K*sizeof(float), cudaMemcpyHostToDevice );
 
    if( cudaErrorCode != SUCCESS ) {
        printf("There was an error copying data from A to A_d. Error Code: %d\n", cudaErrorCode);
    }

    cudaErrorCode = cudaMemcpy( B_d, B, K*N*sizeof(float), cudaMemcpyHostToDevice );
 
    if( cudaErrorCode != SUCCESS ) {
        printf("There was an error copying data from B to B_d. Error Code: %d\n", cudaErrorCode);
    }

// --------------------------------------------END-------------------------------------------------------





    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);



    // TODO

// -------------------------------------------START------------------------------------------------------

    dim3 numThreadsPerBlock( TILE_DIM, TILE_DIM );
    dim3 numBlocks( (N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (M + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y );

    mm_tiled_kernel<<< numBlocks, numThreadsPerBlock>>>( A_d, B_d, C_d, M, N, K);

// --------------------------------------------END-------------------------------------------------------





    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);



    // TODO

// -------------------------------------------START------------------------------------------------------

    cudaErrorCode = cudaMemcpy( C, C_d, M*N*sizeof(float), cudaMemcpyDeviceToHost );

    if( cudaErrorCode != SUCCESS ) {
        printf("There was an error copying data from C_d to C. Error Code: %d\n", cudaErrorCode);
    }

// --------------------------------------------END-------------------------------------------------------





    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);



    // TODO

// -------------------------------------------START------------------------------------------------------

    cudaErrorCode = cudaFree(A_d);
 
    if( cudaErrorCode != SUCCESS ) {
        printf("There was an error deallocating memory from A_d. Error Code: %d\n", cudaErrorCode);
    }

    cudaErrorCode = cudaFree(B_d);
    
    if( cudaErrorCode != SUCCESS ) {
        printf("There was an error deallocating memory from B_d. Error Code: %d\n", cudaErrorCode);
    }
    
    cudaErrorCode = cudaFree(C_d);

    if( cudaErrorCode != SUCCESS ) {
        printf("There was an error deallocating memory from C_d. Error Code: %d\n", cudaErrorCode);
    }

// --------------------------------------------END-------------------------------------------------------





    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");





}

