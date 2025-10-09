#include <algorithm> // For std::swap
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 512

void checkCudaErrors(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// Helper to calculate grid size (number of blocks)
// No changes needed here, but the calculation is more intuitive now.
// For N elements, how many blocks of size T do we need?
inline int calculateGridSize(int N, int threadsPerBlock) {
  // Each thread block processes `threadsPerBlock * 2` elements at the start
  return (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
}

// The kernel remains identical! The optimization is in how we call it.
__global__ void sumArray(const float *input, float *output, int N) {
  __shared__ float sharedData[THREADS_PER_BLOCK];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  // Each thread sums two elements from global memory into one register `sum`
  float sum = 0.0f;
  if (i < N) {
    sum += input[i];
  }
  if (i + blockDim.x < N) {
    sum += input[i + blockDim.x];
  }
  sharedData[tid] = sum;

  __syncthreads();

  // Parallel reduction within the block using shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sharedData[tid] += sharedData[tid + stride];
    }
    __syncthreads();
  }

  // The first thread in each block writes the block's partial sum to the output
  // array
  if (tid == 0)
    output[blockIdx.x] = sharedData[0];
}

int main() {
  int N = 1 << 20; // 1,048,576 elements
  const int threadsPerBlock = THREADS_PER_BLOCK;

  float *h_array = new float[N];
  for (int i = 0; i < N; i++)
    h_array[i] = static_cast<float>(i);

  // CPU reference sum
  double cpuSum = 0.0;
  for (int i = 0; i < N; ++i)
    cpuSum += h_array[i];

  float *d_input, *d_output;
  // Allocate enough space for the original array
  checkCudaErrors(cudaMalloc(&d_input, N * sizeof(float)));

  // The output array size will change, but we need a buffer large enough
  // for the first pass. The number of blocks from the first pass is the max
  // needed.
  int maxBlocks = calculateGridSize(N, threadsPerBlock);
  checkCudaErrors(cudaMalloc(&d_output, maxBlocks * sizeof(float)));

  checkCudaErrors(
      cudaMemcpy(d_input, h_array, N * sizeof(float), cudaMemcpyHostToDevice));

  // ===================================================================
  // OPTIMIZATION: Recursive reduction loop on the GPU
  // ===================================================================
  int numElements = N;
  while (numElements > 1) {
    int gridSize = calculateGridSize(numElements, threadsPerBlock);

    // Launch the kernel to reduce 'numElements' down to 'gridSize' elements
    sumArray<<<gridSize, threadsPerBlock>>>(d_input, d_output, numElements);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(
        cudaGetLastError()); // Good practice to check for kernel launch errors

    // The output of this pass becomes the input for the next pass
    numElements = gridSize;

    // Pointer Swap: Avoid a costly cudaMemcpy by just swapping the pointers.
    // The d_output buffer now holds the new input data for the next iteration.
    std::swap(d_input, d_output);
  }

  // After the loop, the final sum is the first element in the 'd_input' buffer
  // (due to the final swap).
  float gpuSumResult;
  checkCudaErrors(cudaMemcpy(&gpuSumResult, d_input, sizeof(float),
                             cudaMemcpyDeviceToHost));

  double gpuSum = static_cast<double>(gpuSumResult);

  printf("Sum Computed on GPU : %.0f\n", gpuSum);
  printf("Sum Computed on CPU : %.0f\n", cpuSum);

  double epsilon = 1.0e-3;
  if (fabs(gpuSum - cpuSum) > epsilon)
    printf("Test Failed!\n");
  else
    printf("Test Passed!\n");

  delete[] h_array;
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}