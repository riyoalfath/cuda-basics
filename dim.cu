#include <cstdio>

#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1

__global__ void mykernel(void) {
	printf("Hello, I'm a thread in block  %d \n", blockIdx.x);
}

int main() {
  mykernel<<<NUM_BLOCKS,BLOCK_WIDTH>>>();
  
  cudaDeviceSynchronize();
  printf("Done \n");
  return 0;
}
