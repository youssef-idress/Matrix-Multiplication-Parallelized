%%writefile cuda.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void initialise_Matrix(int *a, int n, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            a[i*m+j] = rand() % 100;
        }
    }
}

void print(int* a, int num, int col){
    for(int i = 0; i < num; i++){
        for(int j = 0; j < col; j++){
            printf("%d ", a[i*col + j]);
        }
        printf("\n");
    }
}

__global__ void matrixMultiplicationCuda (int *a, int *b, int *c, int n, int m, int p){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    int temp = 0;

    if((row < n) && (column < p)){
        for(int i = 0; i < m; i++){
            temp += a[row*m+i] * b[i*p+column];
        }
        c[row * p + column] = temp;
    }
}

int main(){
    int num = 1000;
    int col = 2000;
    int p = 1000;

    size_t bytes = num*col*sizeof(int);

    int *host_A, *host_B, *host_C;

    host_A = (int*) malloc(bytes);
    host_B = (int*) malloc(bytes);
    host_C = (int*) malloc(bytes);

    int *Device_A, *Device_B, *Device_C;

    cudaMalloc(&Device_A, bytes);
    cudaMalloc(&Device_B, bytes);
    cudaMalloc(&Device_C, bytes);

    initialise_Matrix(host_A, num, col);
    initialise_Matrix(host_B, col, p);

    cudaMemcpy(Device_A, host_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Device_B, host_B, bytes, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 16;
    int GRID_SIZE = (int)ceil(num/ BLOCK_SIZE);

    dim3 grid(GRID_SIZE,GRID_SIZE);
    dim3 threads(BLOCK_SIZE,BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, NULL);

    matrixMultiplicationCuda <<<grid, threads>>> (Device_A, Device_B, Device_C, num, col, p);

    cudaMemcpy(host_C, Device_C, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("The elapsed time in gpu was %.2f ms\n", milliseconds);
    printf("The number of threads: %d\n", GRID_SIZE * GRID_SIZE * BLOCK_SIZE * BLOCK_SIZE);

    print(host_C, num, p);

    cudaFree(Device_A);
    cudaFree(Device_B);
    cudaFree(Device_C);
    free(host_A);
    free(host_B);
    free(host_C);

    return 0;
}
