#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

__global__ void sum_arr_on_host(float *A, float *B, float *C, const int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
        for (int i = index; i < N; i += stride)
            C[i] = A[i] + B[i];
}

void init_data(float *arr, int size) {
  time_t t;
  srand((unsigned int)time(&t));
  for (int i = 0; i < size; i++) {
    arr[i] = (float)(rand() & 0xFF) / 10.0f;
  }
}

int main(int argc, char **argv) {
    int num_elems = 16000000;
    int num_bytes = num_elems * sizeof(float);
    float *A, *B, *C;
    int blockSize = 256;
    int numBlocks = (num_elems + blockSize -1) / blockSize;
    cudaMallocManaged(&A, num_bytes);
    cudaMallocManaged(&B, num_bytes);
    cudaMallocManaged(&C, num_bytes);
    init_data(A, num_elems);
    init_data(B, num_elems);
    init_data(C, num_elems);
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(A, num_bytes, device, NULL);
    cudaMemPrefetchAsync(B, num_bytes, device, NULL);
    sum_arr_on_host<<<numBlocks, blockSize>>>(A, B, C, num_elems);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}