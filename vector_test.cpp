#include <vector>
#include <iostream>
#include <memory>

using namespace std;

auto main() -> int
{
    const int N = 1 << 20;       // Number of elements in the vectors
    float arrayOfFloat[N];       // A C-style array allocated on the stack
    float *ptrToFloat;           // A pointer to float, not initialized or allocated
    vector<float> vecOfFloat(N); // A C++ vector, dynamically allocated and managed that is initialized with N elements
    vector<float> unassignedVec; // A vector that is declared but not initialized

    unassignedVec.resize(N); // Resize the uninitialized vector to hold N elements

    ptrToFloat = new float[N]; // Dynamically allocate an array of floats
    free(ptrToFloat);          // Incorrectly free the memory allocated with new
    delete[] ptrToFloat;       // Correctly deallocate the memory allocated with new

    ptrToFloat = make_unique<float[]>(N).get(); // Use smart pointer to manage memory
}