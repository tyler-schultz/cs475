#include "MatMultKernel.h"

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

  // Each thread computes one element of Csub in its copy of CValue
  float Avalue1 = 0;
  float Avalue2 = 0;
  float Avalue3 = 0;
  float Avalue4 = 0;

  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
   for (int m = 0;  m < (B.width / FOOTPRINT_SIZE); ++m){
    // Get Bsub and Csub descriptors

	//for(int )
    Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m];
    Csub = &C.elements[C.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col];


    // Copy ELEMENTS OF  ASub and Bsub into shared memory
    // EACH THREAD loads ONE ELEMENT of ASub and ONE of Bsub
    // Notice: it does not need to be the element it requires to
    //         compute its Avalue, as long as all elements are 
    //         collaboratively read. 

    // Notice: every thread declares shared_A and shared_B in shared memory
    //         even though a thread block has only one shared_A and one shared_B
    __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    __shared__ float shared_C[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

    // Each thread copies just one element of shared_B and one element of shared_C
    shared_B[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
    shared_C[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];
    shared_B[thread_row+BLOCK_SIZE][thread_col] = Asub[(thread_row+BLOCK_SIZE )* A.stride + thread_col];
    shared_B[thread_row+BLOCK_SIZE][thread_col+BLOCK_SIZE] = Asub[(thread_row+BLOCK_SIZE )* A.stride + (thread_col+BLOCK_SIZE)];
    shared_B[thread_row][thread_col+BLOCK_SIZE] = Asub[thread_row* A.stride + (thread_col+BLOCK_SIZE)];
    shared_C[thread_row+BLOCK_SIZE][thread_col] = Bsub[(thread_row+BLOCK_SIZE) * B.stride + thread_col];
    shared_C[thread_row][thread_col+BLOCK_SIZE] = Bsub[thread_row * B.stride + (thread_col+BLOCK_SIZE)];
    shared_C[thread_row+BLOCK_SIZE][thread_col+BLOCK_SIZE] = Bsub[(thread_row+BLOCK_SIZE) * B.stride + (thread_col+BLOCK_SIZE)];
    // Synchronize to ensure all elements are read
    __syncthreads();

    // Do an inproduct of one row of shared_A and one col of shared_B
    // computing one Cvalue by accumulation
#pragma unroll
    for(int e=0; e<FOOTPRINT_SIZE; ++e){
       Avalue1 += shared_B[thread_row+BLOCK_SIZE][e] * shared_C[e][thread_col];
       Avalue2 += shared_B[thread_row][e] * shared_C[e][thread_col];
       Avalue3 += shared_B[thread_row][e] * shared_C[e][thread_col+BLOCK_SIZE];
       Avalue4 += shared_B[thread_row+BLOCK_SIZE][e] * shared_C[e][thread_col+BLOCK_SIZE];
    }
    // Synchronize to ensure all Avalues have been incremented
    // before reading in the next shared_B AND shared_C BLOCKS
    __syncthreads();
  }

  // Write Asub to GLOBAL memory.
  // Each thread writes its own cell value.
  Asub[thread_row * A.stride + thread_col] = Avalue2;
  Asub[(thread_row+BLOCK_SIZE) * A.stride +thread_col] = Avalue1;
  Asub[thread_row * A.stride + (thread_col+BLOCK_SIZE)] = Avalue3;
  Asub[(thread_row+BLOCK_SIZE) *A.stride + (thread_col+BLOCK_SIZE)] = Avalue4;
}
