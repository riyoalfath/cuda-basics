#include <cuda_runtime.h>
#include <iostream>

using namespace std;

// __global__ defines this as a CUDA kernel, a function that runs on the GPU
// (device) but is called from the CPU (host). It must have a void return type.
__global__ void hello_kernel() {
  // Each thread has a unique index within its block and grid.
  // threadIdx.x: Thread index within the block
  // blockIdx.x: Block index within the grid
  // blockDim.x: Number of threads in the block
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  // printf can be used from within a kernel for debugging.
  // The output will appear in the host's console.
  printf("Hello from the GPU! BlockIdx: %d, ThreadIdx: %d, BlockDim: %d \n",
         blockIdx.x, threadIdx.x, blockDim.x);
  printf("Hello from the GPU! Global Thread ID: %d\n", id);
}

int main() {

  // Launch the kernel on the device.
  // The <<<...>>> syntax is the execution configuration.
  // The first parameter is the number of blocks in the grid (1).
  // The second parameter is the number of threads per block (1).
  hello_kernel<<<10, 4>>>();

  // Kernel launches are asynchronous. The host (CPU) issues the launch command
  // and continues execution without waiting for the kernel to finish.
  // We must explicitly tell the host to wait for the device to complete
  // all previously issued tasks. Otherwise, main() might exit before
  // the GPU has a chance to print its message.
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize() failed with error: %s\n",
            cudaGetErrorString(err));
  }

  printf("\nKernel execution completed.\n");

  return 0;
}