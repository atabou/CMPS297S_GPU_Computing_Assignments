
#include "common.h"
#include "timer.h"

#define BLOCK_DIM 1024
#define COARSE_FACTOR 4

__global__ void histogram_private_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    __shared__ unsigned int histogram_s[NUM_BINS];

    if( threadIdx.x < NUM_BINS ) {
        histogram_s[threadIdx.x] = 0;
    }

    __syncthreads();

    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if( i < width*height ) {

        unsigned char bin_pos = image[i];

        atomicAdd(&histogram_s[bin_pos], 1);

    }

    __syncthreads();

    if( threadIdx.x < NUM_BINS ) {
        atomicAdd(&bins[threadIdx.x], histogram_s[threadIdx.x]);
    }

}

void histogram_gpu_private(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numBlocks = ( width*height + numThreadsPerBlock - 1)/numThreadsPerBlock;

    histogram_private_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);

}

__global__ void histogram_private_coarse_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    __shared__ int histogram_s[NUM_BINS];

    if( threadIdx.x < NUM_BINS ) {
        histogram_s[threadIdx.x] = 0;
    }

    __syncthreads();

    unsigned int segment = COARSE_FACTOR*blockIdx.x*blockDim.x;

    unsigned int threadSegment = threadIdx.x * COARSE_FACTOR;
    for( int i=0; i < COARSE_FACTOR && segment + threadSegment + i < width*height; i++) {
        unsigned char bin_pos = image[ segment + threadSegment + i  ];
        atomicAdd( &histogram_s[bin_pos], 1 );
    }

    __syncthreads();

    if( threadIdx.x < NUM_BINS ) {
        atomicAdd( &bins[threadIdx.x], histogram_s[threadIdx.x]);
    }

}

void histogram_gpu_private_coarse(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {
    
    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numBlocks = ( width*height + COARSE_FACTOR*numThreadsPerBlock - 1 ) / (COARSE_FACTOR*numThreadsPerBlock);

    histogram_private_coarse_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, bins_d, width, height);


}

