#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "bunch_ann_gpu.h"

#define Z(i,j) Z[i][j]
#define Y(i,j) Y[i][j]
#define I(i,j) I[((i)*(a))+(j)]
#define foo(a,b) b?tanh(a):exp(a)

//Initializes the weights
void initializeW(double* X, long a, long b) {
	for (int i = 0; i < a; ++i) {
		for (int j = 0; j < b; ++j) {
			X[i*a + j] = ((double)rand() / (double)RAND_MAX) * 0.2 - 0.1;
		}
	}
}

// Initializes the inputs
void initializeI(double* X, long a, long b) {
	dim3 threadBlocks(ceil((double)a/2.0), ceil((double)b/2.0));
	initializeIKernel<<<1, threadBlocks>>>(X, a, b);
	cudaDeviceSynchronize();
}

__global__
void initializeIKernel(double* X, long a, long b) {
	double *X_temp = X;
	
	for (int i = 0; i < a; ++i) {
		for (int j = 0; j < b; ++j) {
			X_temp[i*a + j] = j%2;
		}
	}
	
	printf("Finished initializeI\n");
}

// Initializes the outputs
void initializeO(double* X, long a, long b) {
	dim3 threadBlocks(ceil((double)a/2.0), ceil((double)b/2.0));
	initializeOKernel<<<1, threadBlocks>>>(X, a, b);
	cudaDeviceSynchronize();
}

__global__
void initializeOKernel(double* X, long a, long b) {
	double *X_temp = X;
	
	for (int i = 0; i < a; ++i) {
		for (int j = 0; j < b; ++j) {
			X_temp[i*a + j] = i%2;
		}
	}
	
	printf("Finished initializeO\n");
}

// Performs Matrix-Matrix Mulitplication
void mm(double* X, double* Y, double* Z, long A, long B, long C) {
	dim3 threadBlocks(ceil((double)A/2.0), ceil((double)C/2.0));
	mmKernel<<<1, threadBlocks, 2*((A*B)+(B*C))*sizeof(double)>>>(X, Y, Z, A, B, C);
	cudaDeviceSynchronize();
}

__global__
void mmKernel(double *X, double *Y, double *Z, long A, long B, long C) {
	
	extern __shared__ double shared[];
	
	double *X_temp = X;
	double *Y_temp = Y;
	double *Z_temp = Z;
	double *Y_Shared = (double*)&shared[0];
	double *Z_Shared = (double*)&shared[A*B];
	
	// Read into shared Y
	for (int i = 0; i < A; ++i) {
		for (int j = 0; j < B; ++j) {
			Y_Shared[i*A + j] = Y_temp[i*A + j];
		}
	}
		
	// Read into shared Z
	for (int i = 0; i < B; ++i) {
		for (int j = 0; j < C; ++j) {
			if (i < B && j < C) {
				Z_Shared[i*B + j] = Z_temp[i*B + j];
			}
		}
	}
	
	__syncthreads();
	
	// Multiply X = YxZ
	for (int i = 0; i < A; ++i) {
		for (int j = 0; j < B; ++j) {
			for (int k = 0; k < C; ++k) {
				if (j == 0)
					X_temp[i*A + k] = 0;
				X_temp[i*A + k] += Y_Shared[i*A + j] * Z_Shared[j*B + k];
			}
		}
	}
	
	printf("Finished mm\n");
}

// Performs Transposed Matrix- Matrix Mulitplication
void mtm(double* X, double* Y, double* Z, long A, long B, long C) {
	dim3 threadBlocks(ceil((double)A/2.0), ceil((double)C/2.0));
	mtmKernel<<<1, threadBlocks, 2*((A*B)+(B*C))*sizeof(double)>>>(X, Y, Z, A, B, C);
	cudaDeviceSynchronize();
}

// TODO
__global__
void mtmKernel(double *X, double *Y, double *Z, long A, long B, long C) {
	
	extern __shared__ double shared[];
	
	double *X_temp = X;
	double *Y_temp = Y;
	double *Z_temp = Z;
	double *Y_Shared = (double*)&shared[0];
	double *Z_Shared = (double*)&shared[A*B];
	
	// Read into shared Y
	for (int i = 0; i < A; ++i) {
		for (int j = 0; j < B; ++j) {
			Y_Shared[i*A + j] = Y_temp[i*A + j];
		}
	}
		
	// Read into shared Z
	for (int i = 0; i < B; ++i) {
		for (int j = 0; j < C; ++j) {
			if (i < B && j < C) {
				Z_Shared[i*B + j] = Z_temp[i*B + j];
			}
		}
	}
	
	__syncthreads();
	
	// Multiply X = YxZ
	for (int i = 0; i < A; ++i) {
		for (int j = 0; j < B; ++j) {
			for (int k = 0; k < C; ++k) {
				if (j == 0)
					X_temp[i*A + k] = 0;
				X_temp[i*A + k] += Y_Shared[j*B + j] * Z_Shared[j*B + k];
			}
		}
	}
	
	printf("Finished mtm\n");
}

// Performs Matrix-Transposed Matrix Mulitplication
void mmt(double* X, double* Y, double* Z, long A, long B, long C) {
	dim3 threadBlocks(ceil((double)A/2.0), ceil((double)C/2.0));
	mmtKernel<<<1, threadBlocks, 2*((A*B)+(B*C))*sizeof(double)>>>(X, Y, Z, A, B, C);
	cudaDeviceSynchronize();
}

__global__
void mmtKernel(double* X, double* Y, double* Z, long A, long B, long C) {
	
	extern __shared__ double shared[];
	
	double *X_temp = X;
	double *Y_temp = Y;
	double *Z_temp = Z;
	double *Y_Shared = (double*)&shared[0];
	double *Z_Shared = (double*)&shared[A*B];
	
	// Read into shared Y
	for (int i = 0; i < A; ++i) {
		for (int j = 0; j < B; ++j) {
			Y_Shared[i*A + j] = Y_temp[i*A + j];
		}
	}
		
	// Read into shared Z
	for (int i = 0; i < B; ++i) {
		for (int j = 0; j < C; ++j) {
			Z_Shared[i*B + j] = Z_temp[i*B + j];
		}
	}
	
	__syncthreads();
	
	// Multiply X = YxZ
	for (int i = 0; i < A; ++i) {
		for (int j = 0; j < B; ++j) {
			X_temp[i*A + j] = 0;
			for (int k = 0; k < C; ++k) {
				if (j == 0)
					X_temp[i*A + k] = 0;
				X_temp[i*A + j] += Y_Shared[j*B + k] * Z_Shared[i*A + k];
			}
		}
	}
	
	printf("Finished mmt\n");
}

// Performs a point-wise operation
void func(double* X, double* Y, long A, long B, long val) {
	dim3 threadBlocks(ceil((double)A/2.0), ceil((double)B/2.0));
	funcKernel<<<1, threadBlocks, 2*(A*B)*sizeof(double)>>>(X, Y, A, B, val);
	cudaDeviceSynchronize();
}

__global__
void funcKernel(double *X, double *Y, long A, long B, long val) {
	
	extern __shared__ double shared[];
	
	double *X_temp = X;
	double *Y_temp = Y;
	double *Y_Shared = (double*)&shared[0];
	
	// Read into shared Y
	for (int i = 0; i < A; ++i) {
		for (int j = 0; j < B; ++j) {
			Y_Shared[i*A + j] = Y_temp[i*A + j];
		}
	}
	
	__syncthreads();
	
	for (int i = 0; i < A; ++i) {
		for (int j = 0; j < B; ++j) {
			X_temp[i*A + j + val] = foo(Y_Shared[i*A + j], val);
		}
	}
	
	printf("Finished func\n");
}

// Performs a point-wise operation
void gradient_func(double* X, double* Y, long A, long B) {
	dim3 threadBlocks(ceil((double)A/2.0), ceil((double)B/2.0));
	gradient_funcKernel<<<1, threadBlocks, 2*(A*B)*sizeof(double)>>>(X, Y, A, B);
	cudaDeviceSynchronize();
}

__global__
void gradient_funcKernel(double* X, double* Y, long A, long B) {
	
	extern __shared__ double shared[];
	
	double *X_temp = X;
	double *Y_temp = Y;
	double *Y_Shared = (double*)&shared[0];
	
	// Read into shared Y
	for (int i = 0; i < A; ++i) {
		for (int j = 0; j < B; ++j) {
			Y_Shared[i*A + j] = Y_temp[i*A + j];
		}
	}
	
	__syncthreads();
	
	for (int i = 0; i < A; ++i) {
		for (int j = 0; j < B; ++j) {
			X_temp[i*A + j] = Y_Shared[i*A + j + 1] * (1 - pow(tanh(X_temp[i*A + j]), 2));
		}
	}
	
	printf("Finished gradient_func\n");
}

// Calculates the Error
void error(double* X, double* Y, double* Z, long B, long C) {
	dim3 threadBlocks(ceil((double)B/2.0), ceil((double)C/2.0));
	errorKernel<<<1, threadBlocks, 2*(B*C)*sizeof(double)>>>(X, Y, Z, B, C);
	cudaDeviceSynchronize();
}

__global__
void errorKernel(double* X, double* Y, double* Z, long B, long C) {
	extern __shared__ double shared[];
	
	double *X_temp = X;
	double *Y_temp = Y;
	double *Z_temp = Z;
	double *Y_Shared = (double*)&shared[0];
	double *Z_Shared = (double*)&shared[B*C];
	
	// Read into shared Y
	for (int i = 0; i < B; ++i) {
		for (int j = 0; j < C; ++j) {
			Y_Shared[i*B + j] = Y_temp[i*B + j];
		}
	}
		
	// Read into shared Z
	for (int i = 0; i < B; ++i) {
		for (int j = 0; j < C; ++j) {
			if (i < B && j < C) {
				Z_Shared[i*B + j] = Z_temp[i*B + j];
			}
		}
	}
	
	__syncthreads();
	
	for (int i = 0; i < B; ++i) {
		for (int j = 0; j < C; ++j) {
			X_temp[i*B + j] = Y_Shared[i*B + j] - Z_Shared[i*B + j];
		}
	}
	
	printf("Finished error\n");
}

// Performs the summation of probabilities
void reduction(double* Y, double* X, long A, long B) {
	dim3 threadBlocks(ceil((double)A/2.0), ceil((double)B/2.0));
	reductionKernel<<<1, threadBlocks, 2*(A*B)*sizeof(double)>>>(Y, X, A, B);
	cudaDeviceSynchronize();
}

__global__
void reductionKernel(double* Y, double* X, long A, long B) {
	// X was 1D to begin with
	
	extern __shared__ double shared[];
	
	double *X_temp = X;
	double *Y_temp = Y;
	double *Y_Shared = (double*)&shared[0];
	
	// Read into shared Y
	for (int i = 0; i < A; ++i) {
		for (int j = 0; j < B; ++j) {
			Y_Shared[i*A + j] = Y_temp[i*A + j];
		}
	}
	
	__syncthreads();
	
	for (int i = 0; i < A; ++i) {
		X_temp[i] = 0;
		for (int j = 0; j < B; ++j) {
			X_temp[i] += Y_Shared[i*A + j];
		}
	}
	
	printf("Finished reduction\n");
}

// Computes the normalized exponential
void prob(double* Y, double* Z, double* X, long A, long B) {
	dim3 threadBlocks(ceil((double)A/2.0), ceil((double)B/2.0));
	probKernel<<<1, threadBlocks, 2*(A*B)*sizeof(double)>>>(Y, Z, X, A, B);
	cudaDeviceSynchronize();
}

__global__
void probKernel(double* Y, double* Z, double* X, long A, long B) {
	// X was 1D to begin with
	
	extern __shared__ double shared[];
	
	double *X_temp = X;
	double *Y_temp = Y;
	double *Z_temp = Z;
	double *Y_Shared = (double*)&shared[0];
	double *X_Shared = (double*)&shared[A*B];
	
	// Read into shared Y
	for (int i = 0; i < A; ++i) {
		for (int j = 0; j < B; ++j) {
			Y_Shared[i*A + j] = Y_temp[i*A + j];
		}
	}
		
	// Read into shared X
	for (int i = 0; i < A; ++i) {
		X_Shared[i] = X_temp[i];
	}
	
	__syncthreads();
	
	for (int i = 0; i < A; ++i) {
		for (int j = 0; j < B; ++j) {
			Z_temp[i*A + j] = Y_Shared[i*A + j] / X_Shared[i];
		}
	}
	
	printf("Finished prob\n");
}

// Updates the weight matrix
void delta(double* Z, double* Y, long A, long B, double C) {
	dim3 threadBlocks(ceil((double)A/2.0), ceil((double)B/2.0));
	deltaKernel<<<1, threadBlocks, 2*(A*B)*sizeof(double)>>>(Z, Y, A, B, C);
	cudaDeviceSynchronize();
}

__global__
void deltaKernel(double* Z, double* Y, long A, long B, double C) {
	extern __shared__ double shared[];
	
	double *Y_temp = Y;
	double *Z_temp = Z;
	double *Y_Shared = (double*)&shared[0];
	
	// Read into shared Y
	for (int i = 0; i < A; ++i) {
		for (int j = 0; j < B; ++j) {
			Y_Shared[i*A + j] = Y_temp[i*A + j];
		}
	}
	
	__syncthreads();
	
	for (int i = 0; i < A; ++i) {
		for (int j = 0; j < B; ++j) {
			Z_temp[i*A + j] -= C * Y_Shared[i*A + j];
		}
	}
	
	printf("Finished delta\n");
}
