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

__global__ void AddVectors(double** X, double** Y, long A, long B, long val)
{
    int blockStartIndex  = blockIdx.x * blockDim.x * val;
    int threadStartIndex = blockStartIndex + threadIdx.x;
    int threadEndIndex   = blockStartIndex;
    int i;

    for( i=threadStartIndex; i<(blockIdx.x+1)*blockDim.x*N; i+=blockDim.x){
            //need to figure out how to increment this
            X[i][j+val] = foo(Y[i][j],val);    
            
            //what was originally here
            //C[i] = A[i] + B[i];
    }

}
