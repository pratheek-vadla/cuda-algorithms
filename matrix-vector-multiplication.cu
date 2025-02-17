#include <stdio.h>

#define NUM_ROW 1024
#define NUM_COL 1024
#define N 1024 // vector size


// Kernel
__global__ void vector_product(int32_t *A, int32_t *B, int32_t *C)
{
    
}


int main()
{
    size_t mat_size_bytes = NUM_ROW * NUM_COL * sizeof(int32_t);
    size_t vec_size_bytes = N * sizeof(int32_t);

    // host arrays
    int32_t *MAT = (int32_t*)malloc(mat_size_bytes);
    int32_t *B = (int32_t*)malloc(vec_size_bytes);
    int32_t *C = (int32_t*)malloc(vec_size_bytes);

    // device arrays
    int32_t *device_MAT, *device_B, *device_C;
    cudaMalloc(&device_MAT, mat_size_bytes);
    cudaMalloc(&device_B, vec_size_bytes);
    cudaMalloc(&device_C, vec_size_bytes);

    // initialize host arrays
    for (int i = 0; i < NUM_ROW * NUM_COL; i++)
    {
        MAT[i] = 2;   
    }

    for (int i = 0; i < N; i++)
    {
        B[i] = i;
    }

    // copy host arrays to device arrays
    cudaMemcpy(device_MAT, MAT, mat_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B, vec_size_bytes, cudaMemcpyHostToDevice);


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