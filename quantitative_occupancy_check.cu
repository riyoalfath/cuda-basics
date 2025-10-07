#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void sumArray(const float *input, float *output, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    atomicAdd(output, input[idx]);
  }
}

int main(int argc, char const *argv[]) {
  int N = 1 << 20;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("Max threads per multiprocessor: %d\n",
         prop.maxThreadsPerMultiProcessor);

  int minGridSize; // the minimum grid size needed for maximum occupancy
  int blockSize;   // the block size that achieves maximum occupancy

  cudaOccupancyMaxPotentialBlockSize(
      &minGridSize, // output: minimum grid size
      &blockSize,   // output: optimal block size
      sumArray,     // your kernel name
      0,            // dynamic shared memory per block (0 if none)
      0);           // block size limit (0 = no limit)

  int gridSize =
      (N + blockSize - 1) / blockSize; // total number of blocks you’ll launch

  printf("Recommended block size: %d\n", blockSize);
  printf("Minimum grid size: %d\n", minGridSize);
  printf("Computed grid size for N=%d: %d\n", N, gridSize);

  return 0;
}
