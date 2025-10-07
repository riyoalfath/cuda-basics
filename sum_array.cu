#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

void checkCudaErrors(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__global__ void sumArray(const float *input, float *output, int N) {
  __shared__ float sharedData[512];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  float sum = 0.0f;
  if (i < N)
    sum += input[i];
  if (i + blockDim.x < N)
    sum += input[i + blockDim.x];
  sharedData[tid] = sum;

  __syncthreads();

  // Parallel reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sharedData[tid] += sharedData[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0)
    output[blockIdx.x] = sharedData[0];
}

int main() {
  int N = 1 << 20; // 1,048,576 elements
  const int threadsPerBlock = 512;
  const int blocks = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

  float *h_array = new float[N];
  for (int i = 0; i < N; i++)
    h_array[i] = static_cast<float>(i);

  // CPU reference sum
  double cpuSum = 0.0;
  for (int i = 0; i < N; ++i)
    cpuSum += h_array[i];

  float *d_input, *d_output;
  checkCudaErrors(cudaMalloc(&d_input, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_output, blocks * sizeof(float)));

  checkCudaErrors(
      cudaMemcpy(d_input, h_array, N * sizeof(float), cudaMemcpyHostToDevice));

  sumArray<<<blocks, threadsPerBlock>>>(d_input, d_output, N);
  checkCudaErrors(cudaDeviceSynchronize());

  // Retrieve partial sums
  float *partial = new float[blocks];
  checkCudaErrors(cudaMemcpy(partial, d_output, blocks * sizeof(float),
                             cudaMemcpyDeviceToHost));

  double gpuSum = 0.0;
  for (int i = 0; i < blocks; i++)
    gpuSum += partial[i];

  printf("Sum Computed on GPU : %.0f\n", gpuSum);
  printf("Sum Computed on CPU : %.0f\n", cpuSum);

  double epsilon = 1.0e-3;
  if (fabs(gpuSum - cpuSum) > epsilon)
    printf("Test Failed!\n");
  else
    printf("Test Passed!\n");

  delete[] h_array;
  delete[] partial;
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
