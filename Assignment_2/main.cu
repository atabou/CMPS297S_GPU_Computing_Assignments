
#include "common.h"
#include "timer.h"

void mm_cpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
    for (unsigned int row = 0; row < M; ++row) {
        for (unsigned int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for(unsigned int i = 0; i < K; ++i) {
                sum += A[row*K + i]*B[i*N + col];
            }
            C[row*N + col] = sum;
        }
    }
}

int main(int argc, char**argv) {

    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int M = (argc > 1)?(atoi(argv[1])):200;
    unsigned int N = (argc > 2)?(atoi(argv[2])):500;
    unsigned int K = (argc > 3)?(atoi(argv[3])):1000;
    float* A = (float*) malloc(M*K*sizeof(float));
    float* B = (float*) malloc(K*N*sizeof(float));
    float* C_cpu = (float*) malloc(M*N*sizeof(float));
    float* C_gpu = (float*) malloc(M*N*sizeof(float));
    for (unsigned int row = 0; row < M; ++row) {
        for (unsigned int col = 0; col < K; ++col) {
            A[row*K + col] = 1.0*rand()/RAND_MAX;
        }
    }
    for (unsigned int row = 0; row < K; ++row) {
        for (unsigned int col = 0; col < N; ++col) {
            B[row*N + col] = 1.0*rand()/RAND_MAX;
        }
    }

    // Compute on CPU
    startTime(&timer);
    mm_cpu(A, B, C_cpu, M, N, K);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Compute on GPU
    startTime(&timer);
    mm_gpu(A, B, C_gpu, M, N, K);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // Verify result
    for (unsigned int row = 0; row < M; ++row) {
        for (unsigned int col = 0; col < N; ++col) {
            float diff = (C_cpu[row*N + col] - C_gpu[row*N + col])/C_cpu[row*N + col];
            const float tolerance = 0.00001;
            if(diff > tolerance || diff < -tolerance) {
                printf("Mismatch at row %u, col %u (CPU result = %e, GPU result = %e)\n", row, col, C_cpu[row*N + col], C_gpu[row*N + col]);
                exit(0);
            }
        }
    }

    // Free memory
    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);

    return 0;

}

