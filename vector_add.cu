#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

// A helper function for checking CUDA API calls for errors.
void checkCudaErrors(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/**
 * CUDA Kernel for Vector Addition
 * Computes C = A + B for vectors of size N.
 * Each thread computes one element of C.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
  // Calculate the unique global index for this thread.
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check: Since we may launch more threads than N,
  // we must ensure we don't access memory out of bounds.
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  // --- 0. Problem Definition ---
  const int N = 1 << 20; // Number of elements in the vectors (1,048,576)
  size_t size = N * sizeof(float);

  // --- 1. Host Memory Allocation and Initialization ---
  // Use vector for easier memory management on the host.
  float *h_A = new float[N];
  float *h_B = new float[N];
  float *h_C = new float[N];
  float *h_C_ref = new float[N]; // Reference result for verification

  // Initialize host vectors A and B.
  for (int i = 0; i < N; ++i) {
    h_A[i] = sin(i) * sin(i);
    h_B[i] = cos(i) * cos(i);
  }
  // Reference result for verification
  for (int i = 0; i < N; ++i) {
    h_C_ref[i] = h_A[i] + h_B[i];
  }

  // --- 2. Device Memory Allocation ---
  float *d_A, *d_B, *d_C;
  checkCudaErrors(cudaMalloc(&d_A, size));
  checkCudaErrors(cudaMalloc(&d_B, size));
  checkCudaErrors(cudaMalloc(&d_C, size));

  // --- 3. Data Transfer (Host to Device) ---
  printf("Copying data from host to device...");
  checkCudaErrors(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

  // --- 4. Kernel Launch Configuration ---
  // Define the number of threads per block.
  // A multiple of 32 is generally a good choice. 256 is common.
  int threadsPerBlock = 256;

  // Calculate the number of blocks needed in the grid.
  // We use ceiling division to ensure we have enough threads for all elements.
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  printf("Launching kernel with %d blocks of %d threads...\n", blocksPerGrid,
         threadsPerBlock);

  // --- 5. Launch the Kernel ---
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  // Check for any errors launched by the kernel.
  checkCudaErrors(cudaGetLastError());

  // --- 6. Data Transfer (Device to Host) ---
  printf("Copying result from device to host...\n");
  // This cudaMemcpy call also serves as a synchronization point.
  // The host will wait until the kernel is finished before this copy begins.
  checkCudaErrors(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

  // --- 7. Verification and Cleanup ---
  printf("Verifying results...\n");
  float maxError = 0.0f;
  for (int i = 0; i < N; ++i) {
    maxError = fmax(maxError, fabs(h_C[i] - (h_A[i] + h_B[i])));
  }
  printf("Max error: %f\n", maxError);
  // For this specific problem, sin^2(i) + cos^2(i) = 1, so we can also check
  // against 1. for (int i = 0; i < N; i++) maxError = fmax(maxError,
  // fabs(h_C[i] - 1.0f)); cout << "Max error against 1.0: " << maxError <<
  // endl;

  // Free device memory.
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));

  printf("Freeing host memory...\n");

  return 0;
}