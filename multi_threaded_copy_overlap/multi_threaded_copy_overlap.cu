#include <cmath>
#include <cstdio>

#include <omp.h>
#include <algorithm>
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
jacobiKernel(double *d_a_new, const double *d_a,const int iy_start, const int iy_end) {
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    unsigned int row = by * blockDim.y + ty + iy_start;
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

__host__ __inline__ void
mergeMatrices(double *h_a, double *d_a, int chunk_size, int offset, int dev_id, int device_count) {
    if (dev_id == 0) {
        // Copy the first row
        cudaMemcpy(h_a, d_a,
                   WIDTH * sizeof(double),
                   cudaMemcpyDeviceToHost);
    }

    // Copy each chunk based on the device id
    cudaMemcpy(h_a + offset, d_a + WIDTH,
               std::min((WIDTH * HEIGHT) - offset, WIDTH * chunk_size) * sizeof(double),
               cudaMemcpyDeviceToHost);

    if (dev_id == device_count - 1) {
        // Copy the last row
        int lastRow = chunk_size * WIDTH + WIDTH;
        offset = WIDTH * (HEIGHT - 1);
        cudaMemcpy(h_a + offset, d_a + lastRow,
                   WIDTH * sizeof(double),
                   cudaMemcpyDeviceToHost);
    }
}

double *jacobi(int device_count) {
    double *h_a;
    double *d_a_new[MAX_DEVICE];
    int iy_end[MAX_DEVICE];
    cudaEvent_t start, stop;
    float milliseconds = 0;

    if (device_count == 0) {
        cudaGetDeviceCount(&device_count);
    }

    printf("Running with %d GPU(s)\n", device_count);

    h_a = (double *) malloc(WIDTH * HEIGHT * sizeof(double));

#pragma omp parallel num_threads(device_count) shared(h_a)
    {
        // Each thread has its own d_a variable
        double *d_a;

        // As the number of threads is equal to the number of CUDA devices, each thread id
        // can be seen as the device id
        int dev_id = omp_get_thread_num();

        // Set the device for each thread
        cudaSetDevice(dev_id);

        int iy_start;

        int chunk_size;
        int chunk_size_low = (HEIGHT - 2) / device_count;
        int chunk_size_high = chunk_size_low + 1;

        // The number of ranks with a smaller chunk_size
        // Example: HEIGHT = 8192, 2 devices
        // num_ranks_low = 2 * 4095 + 2 - 8190 = 2
        // Devices 0 and 1 with the same chunk size (4095)
        // Example: HEIGHT = 8193, 2 devices
        // num_ranks_low = 2 * 4095 + 2 - 8191 = 1
        // Device 0 with chunk size 4095
        // Device 1 with chunk size 4096
        int num_ranks_low = device_count * chunk_size_low + device_count - (HEIGHT - 2);

        if (dev_id < num_ranks_low)
            chunk_size = chunk_size_low;
        else
            chunk_size = chunk_size_high;

        // Each thread allocates its own d_a and d_a_new[dev_id] with its chunk size
        // and two more rows: top and bottom
        cudaMalloc(&d_a, WIDTH * (chunk_size + 2) * sizeof(double));
        cudaMalloc(d_a_new + dev_id, WIDTH * (chunk_size + 2) * sizeof(double));

        cudaMemset(d_a, 0, WIDTH * (chunk_size + 2) * sizeof(double));
        cudaMemset(d_a_new[dev_id], 0, WIDTH * (chunk_size + 2) * sizeof(double));

        // Calculate local domain boundaries
        int iy_start_global;  // My start index in the global array
        if (dev_id < num_ranks_low) {
            iy_start_global = dev_id * chunk_size_low + 1;
        } else {
            iy_start_global =
                    num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }

        iy_start = 1;
        iy_end[dev_id] = iy_start + chunk_size;

        dim3 dimBlock(TILE_SIZE_X, TILE_SIZE_Y, 1);
        dim3 dimGridDirichlet(1, std::ceil((chunk_size + dimBlock.y - 1) / float(dimBlock.y)), 1);

        // Set dirichlet boundary conditions on left and right border
        dirichlet<<<dimGridDirichlet, dimBlock>>>(d_a, d_a_new[dev_id], chunk_size);
        cudaGetLastError();

        const int top = dev_id > 0 ? dev_id - 1 : (device_count - 1);
        const int bottom = (dev_id + 1) % device_count;

        cudaMemcpy(d_a_new[top] + (iy_end[dev_id] * WIDTH),
                   d_a_new[dev_id] + iy_start * WIDTH, WIDTH * sizeof(double),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_a_new[bottom], d_a_new[dev_id] + (iy_end[dev_id] - 1) * WIDTH,
                   WIDTH * sizeof(double), cudaMemcpyDeviceToDevice);

#pragma omp barrier

#if defined DEBUG && DEBUG == 1
        mergeMatrices(h_a, d_a, chunk_size, iy_start_global * WIDTH, dev_id, device_count);

#pragma omp barrier
#pragma omp master
{
        printf("Initialization\n");
        printMatrix(h_a);
}
#endif

        dim3 dimGrid((WIDTH + dimBlock.x - 1) / dimBlock.x,
                     (chunk_size + dimBlock.y - 1) / dimBlock.y, 1);

#pragma omp master
        {
            // Prepare the timer
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
        }

        for (int i = 0; i < NB_ITERS; i++) {
            jacobiKernel<<<dimGrid, dimBlock>>>(d_a_new[dev_id], d_a, iy_start + 1, iy_end[dev_id] - 1);
            cudaGetLastError();

            jacobiKernel<<<WIDTH / 128 + 1, 128>>>(d_a_new[dev_id], d_a, iy_start, iy_start + 1);
            cudaGetLastError();

            jacobiKernel<<<WIDTH / 128 + 1, 128>>>(d_a_new[dev_id], d_a, iy_end[dev_id] - 1, iy_end[dev_id]);
            cudaGetLastError();

            // Send the first row of the current device to the bottom row of the "top" device
            // Only if the current device isn't already the top one
            if (dev_id > 0) {
                cudaMemcpyAsync(d_a_new[top] + iy_end[top] * WIDTH,
                                d_a_new[dev_id] + (iy_start - 1) * WIDTH,
                                WIDTH * sizeof(double),
                                cudaMemcpyDeviceToDevice);
            }

            // Send the last row of the current device to the top row of the "bottom" device
            // Only if the current device isn't already the bottom one
            if (dev_id < device_count - 1) {
                cudaMemcpyAsync(d_a_new[bottom],
                                d_a_new[dev_id] + (iy_end[dev_id] - 1) * WIDTH,
                                WIDTH * sizeof(double),
                                cudaMemcpyDeviceToDevice);
            }

#pragma omp barrier
            std::swap(d_a_new[dev_id], d_a);
        }


#pragma omp barrier

#pragma omp master
        {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("%d Jacobi iterations done in %lf seconds in a mesh : %dx%d\n", NB_ITERS, milliseconds / 1000, WIDTH,
                   HEIGHT);
        }

        mergeMatrices(h_a, d_a, chunk_size, iy_start_global * WIDTH, dev_id, device_count);

#if defined DEBUG && DEBUG == 1
#pragma omp barrier
#pragma omp master
        {
                printf("Final matrix\n");

                printMatrix(h_a);
        }
#endif

        cudaFree(d_a);
        cudaFree(d_a_new[dev_id]);
    }

    return h_a;
}