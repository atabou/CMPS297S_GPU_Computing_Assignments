
#include "common.h"
#include "timer.h"

void scan_cpu(float* input, float* output, unsigned int N) {
    output[0] = 0.0f;
    for(unsigned int i = 1; i < N; ++i) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

void scan_gpu(float* input, float* output, unsigned int N) {

    Timer timer;

    // Allocate memory
    startTime(&timer);
    float *input_d, *output_d;
    cudaMalloc((void**) &input_d, N*sizeof(float));
    cudaMalloc((void**) &output_d, N*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(input_d, input, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Compute on GPU
    scan_gpu_d(input_d, output_d, N);

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(output, output_d, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free memory
    startTime(&timer);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

int main(int argc, char**argv) {

    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int N = (argc > 1)?(atoi(argv[1])):(1 << 24);
    float* input = (float*) malloc(N*sizeof(float));
    float* output_cpu = (float*) malloc(N*sizeof(float));
    float* output_gpu = (float*) malloc(N*sizeof(float));
    for(unsigned int i = 0; i < N; ++i) {
        input[i] = 1.0*rand()/RAND_MAX;
    }

    // Compute on CPU
    startTime(&timer);
    scan_cpu(input, output_cpu, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Compute on GPU
    startTime(&timer);
    scan_gpu(input, output_gpu, N);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // Verify result
    for(unsigned int i = 0; i < N; ++i) {
        float diff = (output_cpu[i] - output_gpu[i])/output_cpu[i];
        const float tolerance = 0.0001;
        if(diff > tolerance || diff < -tolerance) {
            printf("Mismatch detected at index %u (CPU result = %e, GPU result = %e)\n", i, output_cpu[i], output_gpu[i]);
            exit(0);
        }
    }

    // Free memory
    free(input);

    return 0;

}

