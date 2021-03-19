
#define MASK_RADIUS 2
#define MASK_DIM ((MASK_RADIUS)*2 + 1)

void copyMaskToGPU(float mask[][MASK_DIM]);

void convolution_tiled_gpu(float* input, float* output, unsigned int width, unsigned int height);

