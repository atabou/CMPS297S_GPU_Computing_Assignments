
#include "common.h"
#include "timer.h"

__global__ void histogram_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < width*height) {
        unsigned char b = image[i];
        atomicAdd(&bins[b], 1);
    }
}

void histogram_gpu(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // Call kernel
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (width*height + numThreadsPerBlock - 1)/numThreadsPerBlock;
    histogram_kernel <<< numBlocks, numThreadsPerBlock >>> (image_d, bins_d, width, height);

}

void histogram_cpu(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    for(unsigned int i = 0; i < width*height; ++i) {
        unsigned char b = image[i];
        ++bins[b];
    }
}

void verify(unsigned int* bins_cpu, unsigned int* bins_gpu) {
    for (unsigned int b = 0; b < NUM_BINS; ++b) {
        if(bins_cpu[b] != bins_gpu[b]) {
            printf("Mismatch at bin %u (CPU result = %u, GPU result = %u)\n", b, bins_cpu[b], bins_gpu[b]);
            return;
        }
    }
}

int main(int argc, char**argv) {

    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int height = (argc > 1)?(atoi(argv[1])):4096;
    unsigned int width = (argc > 2)?(atoi(argv[2])):4096;
    unsigned char* image = (unsigned char*) malloc(width*height*sizeof(unsigned char));
    unsigned int* bins_cpu = (unsigned int*) malloc(NUM_BINS*sizeof(unsigned int));
    unsigned int* bins_gpu = (unsigned int*) malloc(NUM_BINS*sizeof(unsigned int));
    for (unsigned int row = 0; row < height; ++row) {
        for (unsigned int col = 0; col < width; ++col) {
            image[row*width + col] = rand()%256;
        }
    }
    memset(bins_cpu, 0, NUM_BINS*sizeof(unsigned int));

    // Compute on CPU
    startTime(&timer);
    histogram_cpu(image, bins_cpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Allocate GPU memory
    startTime(&timer);
    unsigned char *image_d;
    unsigned int *bins_d;
    cudaMalloc((void**) &image_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**) &bins_d, NUM_BINS*sizeof(unsigned int));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(image_d, image, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(bins_d, 0, NUM_BINS*sizeof(unsigned int));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Compute on GPU
    startTime(&timer);
    histogram_gpu(image_d, bins_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time (unoptimized)", GREEN);

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(bins_gpu, bins_d, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Verify result
    verify(bins_cpu, bins_gpu);

    // Compute on GPU
    cudaMemset(bins_d, 0, NUM_BINS*sizeof(unsigned int));
    cudaDeviceSynchronize();
    startTime(&timer);
    histogram_gpu_private(image_d, bins_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time (with privatization and shared memory)", GREEN);

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(bins_gpu, bins_d, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Verify result
    verify(bins_cpu, bins_gpu);

    // Compute on GPU
    cudaMemset(bins_d, 0, NUM_BINS*sizeof(unsigned int));
    cudaDeviceSynchronize();
    startTime(&timer);
    histogram_gpu_private_coarse(image_d, bins_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time (with privatization, shared memory, and thread coarsening)", GREEN);

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(bins_gpu, bins_d, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Verify result
    verify(bins_cpu, bins_gpu);

    // Free GPU memory
    startTime(&timer);
    cudaFree(image_d);
    cudaFree(bins_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

    // Free memory
    free(image);
    free(bins_cpu);
    free(bins_gpu);

    return 0;

}

