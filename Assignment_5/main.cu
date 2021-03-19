
#include "common.h"
#include "timer.h"

float reduce_cpu(float* input, unsigned int N) {
    float sum = 0.0f;
    for(unsigned int i = 0; i < N; ++i) {
        sum += input[i];
    }
    return sum;
}

int main(int argc, char**argv) {

    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int N = (argc > 1)?(atoi(argv[1])):(16000000);
    float* input = (float*) malloc(N*sizeof(float));
    for (unsigned int i = 0; i < N; ++i) {
        input[i] = 1.0*rand()/RAND_MAX;
    }

    // Compute on CPU
    startTime(&timer);
    float sum_cpu = reduce_cpu(input, N);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Compute on GPU
    startTime(&timer);
    float sum_gpu = reduce_gpu(input, N);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // Verify result
    float diff = (sum_cpu - sum_gpu)/sum_cpu;
    const float tolerance = 0.0001;
    if(diff > tolerance || diff < -tolerance) {
        printf("Mismatch detected (CPU result = %e, GPU result = %e)\n", sum_cpu, sum_gpu);
        exit(0);
    }

    // Free memory
    free(input);

    return 0;

}
