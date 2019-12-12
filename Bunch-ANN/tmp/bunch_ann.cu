#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <math.h> 
#include "bunch_ann.h"
//#include "util.h"

#define Z(i,j) Z[i][j]
#define Y(i,j) Y[i][j]
#define I(i,j) I[((i)*(a))+(j)]
#define foo(a,b) b?tanh(a):exp(a)

//Initializes the weights
void initializeW(double* X, long a, long b) {
    long i,j;
    for (i=0; i<a;i++)
        for (j=0; j<b;j++)
            X[i*a + j] = ((double)rand() / (double)RAND_MAX) * 0.2 - 0.1;
}

// Initializes the inputs
void initializeI(double* X, long a, long b) {
	long i,j;
	for (i=0; i<a;i++)
		for (j=0; j<b;j++)
			X[i*a + j] = j%2;
}

// Initializes the outputs
void initializeO(double* X, long a, long b) {
	long i,j;
	for (i=0; i<a;i++)
		for (j=0; j<b;j++)
			X[i*a + j] = i%2;
}

// Performs Matrix-Matrix Mulitplication
void mm(double* X, double* Y, double* Z, long A, long B, long C) {
	/*long i,j,k;
	for (i=0; i<A; i++) 
		for (j=0; j<B; j++)
			for(k=0; k<C; k++) 
			{
				if(j==0) X[i*A + k]=0;
				X[i*A + k] += Y[i*A + j] * Z[j*B + k];
			}*/
	
	dim3 threadBlocks(ceil((double)A/2.0), ceil((double)C/2.0));
	MatMultKernel<<<1, threadBlocks, (A*B)+(B*C)>>>(X, Y, Z, A, B, C);
	cudaDeviceSynchronize();
}

__global__
void MatMultKernel(double *X, double *Y, double *Z, long A, long B, long C) {
	
	int threadDim = blockDim.x; // Contains the dimensions of each thread block as specified by numThreadsInThreadBlocks
	int threadRow = threadIdx.x; // Contains the index of the thread within its thread block
	int threadCol = threadIdx.y;
	
	double *X_temp = X;
	double *Y_temp = Y;
	double *Z_temp = Z;
	double *Y_Shared = (double*)&shared[0];
	double *Z_Shared = (double*)&shared[A*B];
	
	// Read into shared Y
	for (int i = threadRow*threadDim; i < threadRow*threadDim + threadDim; ++i) {
		for (int j = threadCol*threadDim; j < threadCol*threadDim + threadDim; ++j) {
			if (i < A && j < B) {
				Y_Shared[i*A + j] = Y_temp[i*A + j];
			}
		}
	}
	
	// Read into shared Z
	for (int i = threadRow*threadDim; i < threadRow*threadDim + threadDim; ++i) {
		for (int j = threadCol*threadDim; j < threadCol*threadDim + threadDim; ++j) {
			if (i < B && j < C) {
				Z_Shared[i*B + j] = Z_temp[i*B + j];
			}
		}
	}
	
	__syncthreads();
	
	// Multiply AxC = temp
	for (int i = threadRow*threadDim; i < threadRow*threadDim + threadDim; ++i) {
		for (int j = threadCol*threadDim; j < threadCol*threadDim + threadDim; ++j) {
			if (i < A && j < C) {
				float acc = 0.0f;
				for (int a = 0; a < B; ++a) {
					acc += Y_Shared[a*B + j] * Z_Shared[i*B + a];
				}
				X_temp[i*B + j] = acc;
			}
		}
	}
	
	__syncthreads();
	
	
	
	
	/*// matrix blocks
	float *Asub, *Bsub, *Csub;
	// Putting these into registers speeds access.
	int thread_row = threadIdx.y;
	int thread_col = threadIdx.x;
	int block_row = blockIdx.y;
	int block_col = blockIdx.x;
	
	// Each THREAD BLOCK computes one sub matrix Csub of C
	// EACH THREAD creates its own matrix descriptor Csub
	Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];
	
	// Each thread computes one element of Csub in its copy of CValue
	float Cvalue1 = 0;
	float Cvalue2 = 0;
	float Cvalue3 = 0;
	float Cvalue4 = 0;
	
	// Loop over all sub matrices in block_row of A and block_col of B
	// required to compute Csub. Block multiply each pair of sub matrices
	// and accumulate results
	for (int m = 0;  m < (A.width / FOOTPRINT_SIZE); ++m){
	// Get Asub and Bsub descriptors

	//for(int )
	Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m];
	Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col];


	// Copy ELEMENTS OF  ASub and Bsub into shared memory
	// EACH THREAD loads ONE ELEMENT of ASub and ONE of Bsub
	// Notice: it does not need to be the element it requires to
	//         compute its Cvalue, as long as all elements are 
	//         collaboratively read. 

	// Notice: every thread declares shared_A and shared_B in shared memory
	//         even though a thread block has only one shared_A and one shared_B
	__shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
	__shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

	// Each thread copies just one element of shared_A and one element of shared_B
	shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
	shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];
	shared_A[thread_row+BLOCK_SIZE][thread_col] = Asub[(thread_row+BLOCK_SIZE )* A.stride + thread_col];
	shared_A[thread_row+BLOCK_SIZE][thread_col+BLOCK_SIZE] = Asub[(thread_row+BLOCK_SIZE )* A.stride + (thread_col+BLOCK_SIZE)];
	shared_A[thread_row][thread_col+BLOCK_SIZE] = Asub[thread_row* A.stride + (thread_col+BLOCK_SIZE)];
	shared_B[thread_row+BLOCK_SIZE][thread_col] = Bsub[(thread_row+BLOCK_SIZE) * B.stride + thread_col];
	shared_B[thread_row][thread_col+BLOCK_SIZE] = Bsub[thread_row * B.stride + (thread_col+BLOCK_SIZE)];
	shared_B[thread_row+BLOCK_SIZE][thread_col+BLOCK_SIZE] = Bsub[(thread_row+BLOCK_SIZE) * B.stride + (thread_col+BLOCK_SIZE)];
	// Synchronize to ensure all elements are read
	__syncthreads();

	// Do an inproduct of one row of shared_A and one col of shared_B
	// computing one Cvalue by accumulation
	#pragma unroll
	for(int e=0; e<FOOTPRINT_SIZE; ++e){
		Cvalue1 += shared_A[thread_row+BLOCK_SIZE][e] * shared_B[e][thread_col];
		Cvalue2 += shared_A[thread_row][e] * shared_B[e][thread_col];
		Cvalue3 += shared_A[thread_row][e] * shared_B[e][thread_col+BLOCK_SIZE];
		Cvalue4 += shared_A[thread_row+BLOCK_SIZE][e] * shared_B[e][thread_col+BLOCK_SIZE];
	}
	// Synchronize to ensure all Cvalues have been incremented
	// before reading in the next shared_A AND shared_B BLOCKS
	__syncthreads();
	}

	// Write Csub to GLOBAL memory.
	// Each thread writes its own cell value.
	Csub[thread_row * C.stride + thread_col] = Cvalue2;
	Csub[(thread_row+BLOCK_SIZE) * C.stride +thread_col] = Cvalue1;
	Csub[thread_row * C.stride + (thread_col+BLOCK_SIZE)] = Cvalue3;
	Csub[(thread_row+BLOCK_SIZE) *C.stride + (thread_col+BLOCK_SIZE)] = Cvalue4;*/
	
}

void mtm(double* X, double* Y, double* Z, long A, long B, long C) {
	/*Performs Transposed Matrix- Matrix Mulitplication*/
	long i,j,k;
	for (i=0; i<A; i++) 
		for (j=0; j<B; j++)
			for(k=0; k<C; k++)
			{ 
				if(j==0) X[i*A + k]=0;
				X[i*A + k] += Y[j*B + i] * Z[j*B + k];
			}
}

void mmt(double* X, double* Y, double* Z, long A, long B, long C) {
	/*Performs Matrix-Transposed Matrix Mulitplication*/
	long i,j,k;
	for (i=0; i<A; i++) 
		for (j=0; j<B; j++)
		{
			X[i*A + j]=0;
			for(k=0; k<C; k++)
				X[i*A + j] += Y[j*B + k] * Z[i*A + k];
		}
}

void func(double* X, double* Y, long A, long B, long val) {
	/*Performs a point-wise operation*/
	long i,j;
	for (i=0; i<A; i++) 
		for (j=0; j<B; j++)
			X[i*A + j+val] = foo(Y[i*A + j], val); 
}

void gradient_func(double* X, double* Y, long A, long B) {
	/*Performs a point-wise operation*/
	long i,j;
	for (i=0; i<A; i++)
		for (j=0; j<B; j++)  
			X[i*A + j] = Y[i*A + j+1]*(1 - pow(tanh(X[i*A + j]), 2)); 
}

void error(double* X, double* Y, double* Z,  long B, long C) {
	/*Calculates the Error*/
	long i,j;
	for (i=0; i<B; i++)
		for (j=0; j<C; j++)
			X[i*B + j] = Y[i*B + j]-Z[i*B + j]; 
}

void reduction(double* Y, double* X, long A, long B) {
	/*Performs the summation of probabilities*/
	// X was 1D to begin with
	long i,j;
	for (i=0; i<A; i++)
	{
		X[i]=0;
		for (j=0; j<B; j++)
			X[i] += Y[i*A + j];
	}
}

void prob(double* Y,double* Z, double* X, long A, long B) {
	/*Computes the normalized exponential*/
	// X was 1D to begin with
	long i,j;
	for (i=0; i<A; i++)
		for (j=0; j<B; j++)
				Z[i*A + j] = Y[i*A + j] / X[i];
}

void delta(double* Z, double* Y, long A, long B, double C) {
	/*Updates the weight matrix*/
	long i,j;
	for (i=0; i<A; i++)
		for (j=0; j<B; j++) 
			Z[i*A + j] -= C*Y[i*A + j]; 
}
