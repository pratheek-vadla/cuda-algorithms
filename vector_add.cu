#include <stdio.h>

#define N 1024

// Kernel
__global__ void add_vectors(int32_t *A, int32_t *B, int32_t *C)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}


int main()
{
    size_t mem_size = N * sizeof(int32_t);

    // host arrays
    int32_t *A = (int32_t*)malloc(mem_size);
    int32_t *B = (int32_t*)malloc(mem_size);
    int32_t *C = (int32_t*)malloc(mem_size);

    // device arrays
    int32_t *device_A, *device_B, *device_C;
    cudaMalloc(&device_A, mem_size);
    cudaMalloc(&device_B, mem_size);
    cudaMalloc(&device_C, mem_size);

    // initialize host arrays
    for (int i = 0; i < N; i++)
    {
        A[i] = 1;
        B[i] = 2;
    }

    // copy host arrays to device arrays
    cudaMemcpy(device_A, A, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B, mem_size, cudaMemcpyHostToDevice);


    // define threads and blocks for execution
    int num_threads_per_block = 32;
    int num_blocks_per_grid = ceil(N / num_threads_per_block);

    // Launch Kernel
    add_vectors<<<num_blocks_per_grid, num_threads_per_block>>>(device_A, device_B, device_C);

    // copy result from device to host
    cudaMemcpy(C, device_C, mem_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        printf("C[%d]=%d\n", i, C[i]);
    }
    
    free(A);
    free(B);
    free(C);

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    printf("\nThreads Per Block=%d\n", num_threads_per_block);
    printf("\nBlocks Per Grid=%d\n", num_blocks_per_grid);


}