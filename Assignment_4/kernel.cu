
#include "common.h"
#include "timer.h"

#define SUCCESS 0

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(MASK_RADIUS))

__constant__ float mask_c[MASK_DIM][MASK_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {

    // TODO

    unsigned int block_row = OUT_TILE_DIM * blockIdx.y;
    unsigned int block_col = OUT_TILE_DIM * blockIdx.x;

    unsigned int row = block_row + threadIdx.y;
    unsigned int col = block_col + threadIdx.x;

    __shared__ float input_s[IN_TILE_DIM][IN_TILE_DIM];

    if( row < MASK_RADIUS || row >= height + MASK_RADIUS || col < MASK_RADIUS || col >= width + MASK_RADIUS ) {
        input_s[threadIdx.y][threadIdx.x] = 0.0f;
    } else {
        input_s[threadIdx.y][threadIdx.x] = input[ width*(row - MASK_RADIUS) + (col - MASK_RADIUS) ];
    }

    __syncthreads();

    if( threadIdx.x >= MASK_RADIUS && threadIdx.x < OUT_TILE_DIM + MASK_RADIUS && threadIdx.y >= MASK_RADIUS && threadIdx.y < OUT_TILE_DIM + MASK_RADIUS && row < height + MASK_RADIUS && col < width + MASK_RADIUS ) {

        float output_value = 0.0f;

        for( unsigned int i=0; i<MASK_DIM; i++ ) {
           for( unsigned int j=0; j<MASK_DIM; j++) {
                output_value += input_s[threadIdx.y - MASK_RADIUS + i][threadIdx.x - MASK_RADIUS + j] * mask_c[i][j];

            }
        }

        output[ width*(row - MASK_RADIUS) + (col - MASK_RADIUS) ] = output_value;

    }

}


void copyMaskToGPU(float mask[][MASK_DIM]) {

    // Copy mask to constant memory

    int cudaErrorCode = cudaMemcpyToSymbol(mask_c, mask, MASK_DIM * MASK_DIM * sizeof(float));

    if( cudaErrorCode != SUCCESS ) {
        printf("There was an a problem when copying data from the Host to Shared Memory, Error Code: %d", cudaErrorCode);
    }

}

void convolution_tiled_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {

    // Call kernel

    dim3 numThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 numBlocks( (width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM );

    convolution_tiled_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, width, height);



}

