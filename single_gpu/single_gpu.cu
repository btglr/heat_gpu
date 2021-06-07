#include <cmath>
#include <cstdio>
#include "constants.h"
#include "single_gpu.h"

typedef unsigned char bool_t;

__global__ void dirichlet(double *const d_a, double *const d_a_new) {
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    unsigned int col = bx * blockDim.x + tx;
    unsigned int row = by * blockDim.y + ty;

    if (row > HEIGHT || (col != 0 && col != WIDTH - 1))
        return;

    const double y0 = 1;
    d_a[row * WIDTH + col] = y0;
    d_a_new[row * WIDTH + col] = y0;
}

__global__ void
jacobiKernel(double *const d_a_new, const double *d_a, const int iy_start, const int iy_end) {
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    unsigned int row = by * blockDim.y + ty + 1;
    unsigned int col = bx * blockDim.x + tx;

    if (row < iy_end) {
        if (col >= 1 && col < (WIDTH - 1)) {
            const double new_val = 0.25 * (d_a[row * WIDTH + col + 1] + d_a[row * WIDTH + col - 1] +
                                           d_a[(row + 1) * WIDTH + col] + d_a[(row - 1) * WIDTH + col]);
            d_a_new[row * WIDTH + col] = new_val;
        }
    }
}

__host__ __inline__ void printMatrix(double *h_a) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%f ", h_a[y * WIDTH + x]);
        }
        printf("\n");
    }
}

double *jacobi() {
    int i, grid_size_x, grid_size_y;
    double *h_a;
    double *d_a, *d_a_new;
    bool_t evenSteps;
    cudaEvent_t start, stop;
    float milliseconds = 0;

    int iy_start = 1;
    int iy_end = (HEIGHT - 1);

    h_a = (double *) malloc(WIDTH * HEIGHT * sizeof(double));

    cudaMalloc((void **) &d_a, WIDTH * HEIGHT * sizeof(double));
    cudaMalloc((void **) &d_a_new, WIDTH * HEIGHT * sizeof(double));
    cudaMemset(d_a, 0, WIDTH * HEIGHT * sizeof(double));
    cudaMemset(d_a_new, 0, WIDTH * HEIGHT * sizeof(double));

    evenSteps = NB_ITERS % 2 == 0;

    grid_size_x = int(std::ceil((float) WIDTH / TILE_SIZE_X));
    grid_size_y = int(std::ceil((float) HEIGHT / TILE_SIZE_Y));

    dim3 dimGrid(grid_size_x, grid_size_y, 1);
    dim3 dimBlock(TILE_SIZE_X, TILE_SIZE_Y, 1);

    // Prepare the timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    dirichlet<<<dimGrid, dimBlock>>>(d_a, d_a_new);

    cudaMemcpy(h_a, d_a_new, WIDTH * HEIGHT * sizeof(double), cudaMemcpyDeviceToHost);

#if defined DEBUG && DEBUG == 1
    printf("Initialization\n");
    printMatrix(h_a);
#endif

    for (i = 0; i < int(std::ceil((float) NB_ITERS / 2)); i++) {
        jacobiKernel<<<dimGrid, dimBlock>>>(d_a_new, d_a, iy_start, iy_end);

#if defined DEBUG && DEBUG == 1
        printf("Step: %d\n", 2 * i);
        cudaMemcpy(h_a, d_a_new, WIDTH * HEIGHT * sizeof(double), cudaMemcpyDeviceToHost);
        printMatrix(h_a);
#endif

        if (evenSteps || (2 * i + 1) < NB_ITERS) {
            jacobiKernel<<<dimGrid, dimBlock>>>(d_a, d_a_new, iy_start, iy_end);
#if defined DEBUG && DEBUG == 1
            printf("Step: %d\n", 2 * i + 1);
            cudaMemcpy(h_a, d_a, WIDTH * HEIGHT * sizeof(double), cudaMemcpyDeviceToHost);
            printMatrix(h_a);
#endif
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%d Jacobi iterations done in %lf seconds in a mesh : %dx%d\n", NB_ITERS, milliseconds / 1000, WIDTH,
           HEIGHT);

    // Copy data back from the device array to the host array
    if (!evenSteps) {
        cudaMemcpy(h_a, d_a_new, WIDTH * HEIGHT * sizeof(double), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(h_a, d_a, WIDTH * HEIGHT * sizeof(double), cudaMemcpyDeviceToHost);
    }

#if defined DEBUG && DEBUG == 1
    printMatrix(h_a);
#endif

    // Deallocate device arrays
    cudaFree(d_a);
    cudaFree(d_a_new);

    return h_a;
}
