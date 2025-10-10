#include <cstdio>

#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 512

void checkCudaErrors(cudaError_t err) {

  if (err != cudaSuccess) {

    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));

    exit(EXIT_FAILURE);
  }
}

int calculateGridSize(int N, int threadsPerBlock) {

  //   the same as

  //   return ceil(static_cast<double>(N) / threadsPerBlock);

  return (N + threadsPerBlock - 1) / threadsPerBlock;
}

__global__ void sumArray(const double *input, double *output, int N) {

  __shared__ double sharedData[THREADS_PER_BLOCK];

  int tid = threadIdx.x;

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  //   printf("BlockIdx: %d, ThreadIdx: %d, BlockDim: %d, Global Index: %d\n",

  //          blockIdx.x, threadIdx.x, blockDim.x, i);

  double sum = 0.0f;

  if (i < N) {

    sum += input[i];
  }

  sharedData[tid] = sum;

  __syncthreads();

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

  int blocks = calculateGridSize(N, THREADS_PER_BLOCK);

  double *h_array = new double[N];

  for (int i = 0; i < N; i++)

    h_array[i] = static_cast<double>(i);

  double Cpu_sum = 0;

  for (int i = 0; i < N; i++)

    Cpu_sum += h_array[i];

  printf("CPU sum: %f\n", Cpu_sum);

  double *d_input, *d_output;

  checkCudaErrors(cudaMalloc(&d_input, N * sizeof(double)));

  checkCudaErrors(cudaMalloc(&d_output, blocks * sizeof(double)));

  checkCudaErrors(

      cudaMemcpy(d_input, h_array, N * sizeof(double), cudaMemcpyHostToDevice));

  sumArray<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);

  checkCudaErrors(cudaDeviceSynchronize());

  double *h_output = new double[blocks];

  checkCudaErrors(cudaMemcpy(h_output, d_output, blocks * sizeof(double),

                             cudaMemcpyDeviceToHost));

  double Gpu_sum = 0;

  for (int i = 0; i < blocks; i++)

    Gpu_sum += h_output[i];

  printf("GPU sum: %f\n", Gpu_sum);

  checkCudaErrors(cudaFree(d_input));

  checkCudaErrors(cudaFree(d_output));

  delete[] h_array;

  delete[] h_output;

  return 0;
}
