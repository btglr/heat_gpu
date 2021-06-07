#include <cmath>
#include <cstdio>
#include <iostream>
#include "constants.h"

__global__ void dirichlet(double *const d_a, double *const d_a_new, int chunk_size) {
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    unsigned int col = bx * blockDim.x + tx;
    unsigned int row = by * blockDim.y + ty;

    // Exit out of the thread if the row is greater than the chunk size
    // Or if the column is greater than 0, because we only need to fill two columns
    // And we use the same thread to write both
    if (row > chunk_size + 1 || col > 0)
        return;

    const double y0 = 1;
    d_a[row * WIDTH] = y0;
    d_a[row * WIDTH + (WIDTH - 1)] = y0;
    d_a_new[row * WIDTH] = y0;
    d_a_new[row * WIDTH + (WIDTH - 1)] = y0;
}

__global__ void
jacobiKernel(double *d_a_new, const double *d_a, const int iy_start, const int iy_end, double *d_a_new_top,
             double *d_a_new_bottom, const int top_chunk_size, const int dev_id, const int device_count) {
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

            // Set the value of the top device's cell if the current device isn't already the top one
            if (iy_start == row && dev_id != 0) {
                d_a_new_top[top_chunk_size * WIDTH + col] = new_val;
            }

            // Set the value of the bottom device's cell if the current device isn't already the bottom one
            if ((iy_end - 1) == row && dev_id != (device_count - 1)) {
                d_a_new_bottom[col] = new_val;
            }
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

__host__ __inline__ void mergeMatrices(double *h_a, double **d_a, int *chunk_size, int device_count) {
    int offset = WIDTH;

    // Copy the first row
    cudaMemcpy(h_a, d_a[0],
               WIDTH * sizeof(double),
               cudaMemcpyDeviceToHost);

    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
        // Copy each chunk based on the device id
        cudaMemcpy(h_a + offset, d_a[dev_id] + WIDTH,
                   std::min((WIDTH * HEIGHT) - offset, WIDTH * chunk_size[dev_id]) * sizeof(double),
                   cudaMemcpyDeviceToHost);

        offset += std::min(chunk_size[dev_id] * WIDTH, (WIDTH * HEIGHT) - offset);
    }

    // Copy the last row
    int lastRow = chunk_size[device_count - 1] * WIDTH + WIDTH;
    offset = WIDTH * (HEIGHT - 1);
    cudaMemcpy(h_a + offset, d_a[device_count - 1] + lastRow,
               WIDTH * sizeof(double),
               cudaMemcpyDeviceToHost);
}

double *jacobi(int device_count) {
    int i, dev_id;
    double *h_a;
    double *d_a[MAX_DEVICE];
    double *d_a_new[MAX_DEVICE];
    cudaEvent_t start, stop;
    float milliseconds = 0;

    if (device_count == 0) {
        cudaGetDeviceCount(&device_count);
    }

    printf("Running with %d GPU(s)\n", device_count);

    int iy_start[MAX_DEVICE];
    int iy_end[MAX_DEVICE];

    int chunk_size[MAX_DEVICE];

    dim3 dimBlock(TILE_SIZE_X, TILE_SIZE_Y, 1);

    h_a = (double *) malloc(WIDTH * HEIGHT * sizeof(double));

    for (dev_id = 0; dev_id < device_count; dev_id++) {
        cudaSetDevice(dev_id);

        int chunk_size_low = (HEIGHT - 2) / device_count;
        int chunk_size_high = chunk_size_low + 1;

        // Number of ranks with chunk_size = chunk_size_low
        int num_ranks_low = device_count * chunk_size_low + device_count - (HEIGHT - 2);
        if (dev_id < num_ranks_low)
            chunk_size[dev_id] = chunk_size_low;
        else
            chunk_size[dev_id] = chunk_size_high;

        cudaMalloc(d_a + dev_id, WIDTH * (chunk_size[dev_id] + 2) * sizeof(double));
        cudaMalloc(d_a_new + dev_id, WIDTH * (chunk_size[dev_id] + 2) * sizeof(double));

        cudaMemset(d_a[dev_id], 0, WIDTH * (chunk_size[dev_id] + 2) * sizeof(double));
        cudaMemset(d_a_new[dev_id], 0, WIDTH * (chunk_size[dev_id] + 2) * sizeof(double));

        iy_start[dev_id] = 1;
        iy_end[dev_id] = iy_start[dev_id] + chunk_size[dev_id];

        // Set dirichlet boundary conditions on left and right border
        dim3 dimGrid(1, std::ceil((chunk_size[dev_id] + dimBlock.y - 1) / float(dimBlock.y)), 1);
        dirichlet<<<dimGrid, dimBlock>>>(d_a[dev_id], d_a_new[dev_id], chunk_size[dev_id]);
        cudaGetLastError();
        cudaDeviceSynchronize();
    }

    for (dev_id = 0; dev_id < device_count; ++dev_id) {
        cudaSetDevice(dev_id);

        const int top = dev_id > 0 ? dev_id - 1 : (device_count - 1);
        const int bottom = (dev_id + 1) % device_count;

        int canAccessPeer = 0;

        cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, top);

        if (canAccessPeer) {
            cudaDeviceEnablePeerAccess(top, 0);
        }

        if (top != bottom) {
            cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, bottom);

            if (canAccessPeer) {
                cudaDeviceEnablePeerAccess(bottom, 0);
            }
        }
    }

    // Prepare the timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (dev_id = 0; dev_id < device_count; ++dev_id) {
        cudaSetDevice(dev_id);
        cudaDeviceSynchronize();
    }

#if defined DEBUG && DEBUG == 1
    printf("Initialization\n");

    mergeMatrices(h_a, d_a, chunk_size, device_count);
    printMatrix(h_a);
#endif

    cudaEventRecord(start);

    for (i = 0; i < NB_ITERS; i++) {
        for (dev_id = 0; dev_id < device_count; ++dev_id) {
            const int top = dev_id > 0 ? dev_id - 1 : (device_count - 1);
            const int bottom = (dev_id + 1) % device_count;

            cudaSetDevice(dev_id);

            dim3 dimGrid((WIDTH + dimBlock.x - 1) / dimBlock.x,
                         (chunk_size[dev_id] + dimBlock.y - 1) / dimBlock.y, 1);

            jacobiKernel<<<dimGrid, dimBlock>>>(d_a_new[dev_id], d_a[dev_id], iy_start[dev_id], iy_end[dev_id],
                                                d_a_new[top], d_a_new[bottom], iy_end[top], dev_id, device_count);
            std::swap(d_a_new[dev_id], d_a[dev_id]);
        }
    }

    for (dev_id = 0; dev_id < device_count; ++dev_id) {
        cudaSetDevice(dev_id);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%d Jacobi iterations done in %lf seconds in a mesh : %dx%d\n", NB_ITERS, milliseconds / 1000, WIDTH,
           HEIGHT);

    mergeMatrices(h_a, d_a, chunk_size, device_count);

#if defined DEBUG && DEBUG == 1
    printf("Final matrix\n");

    printMatrix(h_a);
#endif

    // Deallocate device arrays
    for (dev_id = (device_count - 1); dev_id >= 0; --dev_id) {
        cudaSetDevice(dev_id);
        cudaFree(d_a[dev_id]);
        cudaFree(d_a_new[dev_id]);
    }

    return h_a;
}