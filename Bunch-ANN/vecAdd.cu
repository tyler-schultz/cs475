///
/// vecAddKernel01.cu
/// For CSU CS575 Spring 2011
/// Instructor: Wim Bohm
/// Based on code from the CUDA Programming Guide
/// By David Newman
/// Created: 2011-02-16
/// Last Modified: 2011-02-16 DVN
///
/// This Kernel adds two Vectors A and B in C on GPU
/// without using coalesced memory access.
/// 

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int blockStartIndex  = blockIdx.x * blockDim.x*N;
    int threadStartIndex = blockStartIndex + threadIdx.x;
    int threadEndIndex   = blockStartIndex;
    int i;

    for( i=threadStartIndex; i<(blockIdx.x+1)*blockDim.x*N; i+=blockDim.x){
                C[i] = A[i] + B[i];
    }

}
~   
