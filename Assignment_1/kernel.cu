
#include "common.h"
#include "timer.h"

#define SUCCESS 0


__global__ void vecMax_kernel(double* a, double* b, double* c, unsigned int M) {

	int thread_pos = blockDim.x * blockIdx.x + threadIdx.x;

	if( thread_pos < M ) {
		c[thread_pos] = a[thread_pos] > b[thread_pos] ? a[thread_pos] : b[thread_pos];
	}

}


void vecMax_gpu(double* a, double* b, double* c, unsigned int M) {

    int cudaErrorCode = SUCCESS;

    Timer timer;

    Allocate GPU memory
    startTime(&timer);

    double *a_d, *b_d, *c_d;

    cudaErrorCode = cudaMalloc((void**) &a_d, M*sizeof(double));
	
    if( cudaErrorCode != SUCCESS ) {
        printf("There was a problem allocating memory for the array a_d on the GPU. Error Code: %d", cudaErrorCode);
        printf("\n\n");
    }

    cudaErrorCode = cudaMalloc((void**) &b_d, M*sizeof(double));

    if( cudaErrorCode != SUCCESS ) {
        printf("There was a problem allocating memory for the array b_d on the GPU. Error Code: %d", cudaErrorCode);
        printf("\n\n");
    }

	cudaErrorCode = cudaMalloc((void**) &c_d, M*sizeof(double));

    if( cudaErrorCode != SUCCESS ) {
        printf("There was a problem allocating memory for the array c_d on the GPU. Error Code: %d", cudaErrorCode);
        printf("\n\n");
    }

	cudaDeviceSynchronize();
	stopTime(&timer);
	printElapsedTime(timer, "Allocation time");

	// Copy data to GPU
	startTime(&timer);

	cudaErrorCode = cudaMemcpy(a_d, a, M*sizeof(double), cudaMemcpyHostToDevice);

    if( cudaErrorCode != SUCCESS ) {
        printf("There was a problem copying a's data to the GPU. Error Code: %d", cudaErrorCode);
        printf("\n\n");
    }

	cudaErrorCode = cudaMemcpy(b_d, b, M*sizeof(double), cudaMemcpyHostToDevice);

    if( cudaErrorCode != SUCCESS ) {
        printf("There was a problem copying b's data to the GPU. Error Code: %d", cudaErrorCode);
        printf("\n\n");
    }

	cudaDeviceSynchronize();
	stopTime(&timer);
	printElapsedTime(timer, "Copy to GPU time");

	// Call kernel
	startTime(&timer);

	const unsigned int numThreadsPerBlock = 512;
	const unsigned int numBlocks = (M + numThreadsPerBlock - 1) /numThreadsPerBlock;
	vecMax_kernel <<< numBlocks, numThreadsPerBlock >>> (a_d, b_d, c_d, M);

	cudaDeviceSynchronize();
	stopTime(&timer);
	printElapsedTime(timer, "Kernel time", GREEN);

	// Copy data from GPU
	startTime(&timer);

	cudaErrorCode = cudaMemcpy(c, c_d, M*sizeof(double), cudaMemcpyDeviceToHost);

    if( cudaErrorCode != SUCCESS ) {
        printf( "There was an error copying c_d's data from the GPU. Error Code: %d", cudaErrorCode);
        printf("\n\n");
    }

	cudaDeviceSynchronize();
	stopTime(&timer);
	printElapsedTime(timer, "Copy from GPU time");

	// Free GPU memory
	startTime(&timer);

	cudaErrorCode = cudaFree( a_d );
	
    if( cudaErrorCode != SUCCESS ) {
        printf( "There was an error deallocating a_d's memory on the GPU. Error Code: %d", cudaErrorCode);
        printf("\n\n");
    }

    cudaErrorCode = cudaFree( b_d );

    if( cudaErrorCode != SUCCESS ) {
        printf( "There was an error deallocating b_d's memory on the GPU. Error Code: %d", cudaErrorCode);
        printf("\n\n");
    }
	
    cudaErrorCode = cudaFree( c_d );

    if( cudaErrorCode != SUCCESS ) {
        printf( "There was an error deallocating c_d's memory on the GPU. Error Code: %d", cudaErrorCode);
        printf("\n\n");
    }



	cudaDeviceSynchronize();
	stopTime(&timer);
	printElapsedTime(timer, "Deallocation time");

}

