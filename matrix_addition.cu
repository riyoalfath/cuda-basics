#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CPU function for Matrix Addition
void matrixAddCPU(const float *A, const float *B, float *C, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int idx = i * N + j;
      C[idx] = A[idx] + B[idx];
    }
  }
}

// CUDA Kernel for 2D Matrix Addition
__global__ void matrixAdd(const float *A, const float *B, float *C, int M,
                          int N) {
  // Calculate the global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check
  if (row < M && col < N) {
    int idx = row * N + col;
    C[idx] = A[idx] + B[idx];
  }
}

int main() {
  // Matrix dimensions
  int M = 1024;
  int N = 1024;
  int numElements = M * N;
  size_t size = numElements * sizeof(float);

  // Host matrices using malloc
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C_gpu = (float *)malloc(size);
  float *h_C_cpu = (float *)malloc(size);

  // Initialize host matrices
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = (float)rand() / RAND_MAX;
    h_B[i] = (float)rand() / RAND_MAX;
  }

  // --- CPU Execution and Timing ---
  clock_t start_cpu = clock();
  matrixAddCPU(h_A, h_B, h_C_cpu, M, N);
  clock_t end_cpu = clock();
  double cpu_duration_ms =
      1000.0 * (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
  printf("CPU execution time: %f ms\n", cpu_duration_ms);

  // --- GPU Execution and Timing ---
  // Device pointers
  float *d_A, *d_B, *d_C;

  // Allocate memory on the device
  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Define thread block and grid dimensions
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // Use CUDA events for accurate GPU timing
  cudaEvent_t start_gpu, stop_gpu;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);

  // Launch the kernel
  cudaEventRecord(start_gpu);
  matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N);
  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);

  float gpu_duration_ms = 0;
  cudaEventElapsedTime(&gpu_duration_ms, start_gpu, stop_gpu);
  printf("GPU kernel execution time: %f ms\n", gpu_duration_ms);

  // Copy the result back from device to host
  cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

  // --- Verification ---
  double error = 0.0;
  for (int i = 0; i < numElements; ++i) {
    error += fabs(h_C_cpu[i] - h_C_gpu[i]);
  }
  printf("Total difference between CPU and GPU results: %f\n", error);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start_gpu);
  cudaEventDestroy(stop_gpu);

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C_gpu);
  free(h_C_cpu);

  return 0;
}