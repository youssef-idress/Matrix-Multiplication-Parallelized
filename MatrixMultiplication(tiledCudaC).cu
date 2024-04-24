%%writefile tiled.cu
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define TILE_SIZE 32

__global__ void tiled(int *a, int *b, int *c, int n, int m){
    __shared__ int A[TILE_SIZE][TILE_SIZE];
    __shared__ int B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    int temp = 0;

    // Loop over tiles of matrix A and B
    for(int i = 0; i < (m / TILE_SIZE); i++){
        // Load tile of matrix A into shared memory
        A[ty][tx] = a[row * m + i * TILE_SIZE + tx];
        // Load tile of matrix B into shared memory
        B[ty][tx] = b[(i * TILE_SIZE + ty) * n + col];
        __syncthreads();

        // Compute partial sum for the tile
        for(int j = 0; j < TILE_SIZE; j++){
            temp += A[ty][j] * B[j][tx];
        }

        // Wait for all threads to finish using shared memory
        __syncthreads();
    }

    // Write result to global memory
    if(row < n && col < m)
        c[row * m + col] = temp;
}


void initialise_Matrix(int *a, int n, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            a[i * m + j] = rand() % 100;
        }
    }
}

void printMatrix(int *a, int n, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            printf("%d ", a[i * m + j]);
        }
        printf("\n");
    }
}

int main(){
    int n = 1000;
    int m = 2000;

    size_t bytes = n * m * sizeof(int);

    int *host_A, *host_B, *host_C;

    host_A = (int *) malloc(bytes);
    host_B = (int *) malloc(bytes);
    host_C = (int *) malloc(bytes);

    int *Device_A, *Device_B, *Device_C;

    cudaMalloc(&Device_A, bytes);
    cudaMalloc(&Device_B, bytes);
    cudaMalloc(&Device_C, bytes);

    initialise_Matrix(host_A, n, m);
    initialise_Matrix(host_B, n, m);

    cudaMemcpy(Device_A, host_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Device_B, host_B, bytes, cudaMemcpyHostToDevice);

    dim3 grid((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    dim3 threads(TILE_SIZE, TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, NULL);

    tiled<<<grid, threads>>>(Device_A, Device_B, Device_C, n, m);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    cudaMemcpy(host_C, Device_C, bytes, cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("The elapsed time in gpu was %.2f ms\n", milliseconds);

    printf("Matrix C:\n");
    printMatrix(host_C, n, m);

    free(host_A);
    free(host_B);
    free(host_C);

    cudaFree(Device_A);
    cudaFree(Device_B);
    cudaFree(Device_C);

    return 0;
}
