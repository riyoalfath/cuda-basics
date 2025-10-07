#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

// A helper function for checking CUDA API calls for errors.
void checkCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * CUDA Kernel for Vector Addition
 * Computes C = A + B for vectors of size N.
 * Each thread computes one element of C.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    // Calculate the unique global index for this thread.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: Since we may launch more threads than N,
    // we must ensure we don't access memory out of bounds.
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    // --- 0. Problem Definition ---
    const int N = 1 << 20; // Number of elements in the vectors (1,048,576)
    size_t size = N * sizeof(float);

    // --- 1. Host Memory Allocation and Initialization ---
    // Use vector for easier memory management on the host.
    vector<float> h_A(N);
    vector<float> h_B(N);
    vector<float> h_C(N);
    vector<vector<float>> h_C_ref(N); // Reference result for verification

    // Initialize host vectors A and B.
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = sin(i) * sin(i);
        h_B[i] = cos(i) * cos(i);
    }
    // Reference result for verification
    for (int i = 0; i < N; ++i)
    {
        h_C_ref[i] = h_A[i] + h_B[i];
    }

    // --- 2. Device Memory Allocation ---
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc(&d_A, size));
    checkCudaErrors(cudaMalloc(&d_B, size));
    checkCudaErrors(cudaMalloc(&d_C, size));

    // --- 3. Data Transfer (Host to Device) ---
    cout << "Copying data from host to device..." << endl;
    checkCudaErrors(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    // --- 4. Kernel Launch Configuration ---
    // Define the number of threads per block.
    // A multiple of 32 is generally a good choice. 256 is common.
    int threadsPerBlock = 256;

    // Calculate the number of blocks needed in the grid.
    // We use ceiling division to ensure we have enough threads for all elements.
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cout << "Launching kernel with " << blocksPerGrid << " blocks of "
         << threadsPerBlock << " threads each." << endl;

    // --- 5. Launch the Kernel ---
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for any errors launched by the kernel.
    checkCudaErrors(cudaGetLastError());

    // --- 6. Data Transfer (Device to Host) ---
    cout << "Copying result from device to host..." << endl;
    // This cudaMemcpy call also serves as a synchronization point.
    // The host will wait until the kernel is finished before this copy begins.
    checkCudaErrors(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    // --- 7. Verification and Cleanup ---
    cout << "Verifying result..." << endl;
    float maxError = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        maxError = fmax(maxError, fabs(h_C[i] - (h_A[i] + h_B[i])));
    }
    cout << "Max error: " << maxError << endl;
    // For this specific problem, sin^2(i) + cos^2(i) = 1, so we can also check against 1.
    // for (int i = 0; i < N; i++) maxError = fmax(maxError, fabs(h_C[i] - 1.0f));
    // cout << "Max error against 1.0: " << maxError << endl;

    // Free device memory.
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    cout << "Test PASSED" << endl;

    return 0;
}