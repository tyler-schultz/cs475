#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <sys/errno.h>
#include <omp.h>

void MMScan(float ***X, float ***Y, long start, long end, long size) {
    long n, i, j, k;
    for (i = 0; i < size; ++i) {
        for (j = 0; j < size; ++j) {
            Y[start][i][j] = X[start][i][j];
        }
    }
#ifdef FAUX  // incorrect parallelization 
#pragma omp parallel for
#endif // incorrect parallelization 
    for (n = start+1; n <= end; ++n) {
        for (i = 0; i < size; ++i) {
            for (j = 0; j < size; ++j) {
                float acc = 0;
                for (k = 0; k < size; ++k) {
                    acc = acc + Y[n-1][i][k] * X[n][k][j];
                }
                Y[n][i][j] = acc;
            }
        }
    }
}

__global__
void phase1(float *x, float *r1, long N, long B) {

    int threadDim = blockDim.x; // Contains the dimensions of each thread block as specified by numThreadsInThreadBlocks
    int threadRow = threadIdx.x; // Contains the index of the thread within its thread block
    int threadCol = threadIdx.y;
    int g = blockIdx.x; // Contains the thread block within the grid
    int G = gridDim.x; // Contains the dimensions of the grid as specified by numThreadBlocks
    int n = N / G;
    
    // Read the matrix into shared memory.
    extern __shared__ float shared[];
    
    float *x_temp = x;
    float *r1_temp = r1;
    float *A = (float*)&shared[0];
    float *C = (float*)&shared[B*B];
    float *temp = (float*)&shared[2*B*B];

    // Read into shared A
    for (int j = threadRow*threadDim; j < threadRow*threadDim + threadDim; ++j) {
        for (int k = threadCol*threadDim; k < threadCol*threadDim + threadDim; ++k) {
            A[j*B + k] = x_temp[(g*n)*B*B + j*B + k];
        }
    }
    
    for (int i = 1; i < n; ++i) {
        
        // Read into shared C
        for (int j = threadRow*threadDim; j < threadRow*threadDim + threadDim; ++j) {
            for (int k = threadCol*threadDim; k < threadCol*threadDim + threadDim; ++k) {
                C[j*B + k] = x_temp[(g*n+i)*B*B + j*B + k];
            }
        }
        __syncthreads();
        
        // Multiply AxC = temp
        for (int j = threadRow*threadDim; j < threadRow*threadDim + threadDim; ++j) {
            for (int k = threadCol*threadDim; k < threadCol*threadDim + threadDim; ++k) {
                float acc = 0.0f;
                for (int a = 0; a < B; ++a) {
                    acc += A[a*B + k] * C[j*B + a];
                }
                temp[j*B + k] = acc;
            }
        }
        
        // Copy temp back to A
        for (int j = threadRow*threadDim; j < threadRow*threadDim + threadDim; ++j) {
            for (int k = threadCol*threadDim; k < threadCol*threadDim + threadDim; ++k) {
                A[j*B + k] = temp[j*B + k];
            }
        }
        
    }
    
    __syncthreads();
    
    // Copy back to R1
    for (int j = threadRow*threadDim; j < threadRow*threadDim + threadDim; ++j) {
        for (int k = threadCol*threadDim; k < threadCol*threadDim + threadDim; ++k) {
            r1_temp[g*B*B + j*B + k] = temp[j*B + k];
        }
    }
    
}

__global__
void phase2(float *r1, float *r2, long G, long B) {
    
    int threadDim = blockDim.x; // Contains the dimensions of each thread block as specified by numThreadsInThreadBlocks
    int threadRow = threadIdx.x; // Contains the index of the thread within its thread block
    int threadCol = threadIdx.y;
    
    // Read the matrix into shared memory.
    extern __shared__ float shared[];
    
    float *r1_temp = r1;
    float *r2_temp = r2;
    float *A = (float*)&shared[0];
    float *C = (float*)&shared[B*B];

    // Read into shared A
    for (int j = threadRow*threadDim; j < threadRow*threadDim + threadDim; ++j) {
        for (int k = threadCol*threadDim; k < threadCol*threadDim + threadDim; ++k) {
            r2_temp[1*B*B + j*B + k] = A[j*B + k] = r1_temp[0 + j*B + k];
        }
    }
    
    for (int i = 2; i < G; ++i) {
        
        // Read into shared C
        for (int j = threadRow*threadDim; j < threadRow*threadDim + threadDim; ++j) {
            for (int k = threadCol*threadDim; k < threadCol*threadDim + threadDim; ++k) {
                C[j*B + k] = r1_temp[(i-1)*B*B + j*B + k];
            }
        }
        __syncthreads();
        
        // Multiply AxC = r2[i]
        for (int j = threadRow*threadDim; j < threadRow*threadDim + threadDim; ++j) {
            for (int k = threadCol*threadDim; k < threadCol*threadDim + threadDim; ++k) {
                float acc = 0.0f;
                for (int a = 0; a < B; ++a) {
                    acc += A[a*B + k] * C[j*B + a];
                }
                r2_temp[i*B*B + j*B + k] = acc;
            }
        }
        
        __syncthreads();
        
        // Copy r2[i] back to A
        for (int j = threadRow*threadDim; j < threadRow*threadDim + threadDim; ++j) {
            for (int k = threadCol*threadDim; k < threadCol*threadDim + threadDim; ++k) {
                A[j*B + k] = r2_temp[i*B*B + j*B + k];
            }
        }
        
    }
}

__global__
void phase3(float *x, float *r2, float *y, long N, long B) {
    
    int threadDim = blockDim.x; // Contains the dimensions of each thread block as specified by numThreadsInThreadBlocks
    int threadRow = threadIdx.x; // Contains the index of the thread within its thread block
    int threadCol = threadIdx.y;
    int g = blockIdx.x; // Contains the thread block within the grid
    int G = gridDim.x; // Contains the dimensions of the grid as specified by numThreadBlocks
    int n = N / G;
    
    // Read the matrix into shared memory.
    extern __shared__ float shared[];
    
    float *x_temp = x;
    float *r2_temp = r2;
    float *y_temp = y;
    float *A = (float*)&shared[0];
    float *C = (float*)&shared[B*B];
    float *T = (float*)&shared[2*B*B];

    // Read each element of r2 into the full block of y
    for (int i = n*g; i < n*g + n; ++i) {
        for (int j = threadRow*threadDim; j < threadRow*threadDim + threadDim; ++j) {
            for (int k = threadCol*threadDim; k < threadCol*threadDim + threadDim; ++k) {
                y_temp[i*B*B + j*B + k] = r2_temp[g*B*B + j*B + k];
            }
        }
    }
    __syncthreads();
    
    for (int yi = n*g; yi < n*g + n; ++yi) {
        
        __syncthreads();
        // Read into shared A, X[0]xR2[0]
        for (int j = threadRow*threadDim; j < threadRow*threadDim + threadDim; ++j) {
            for (int k = threadCol*threadDim; k < threadCol*threadDim + threadDim; ++k) {
                if (g == 0) {
                    A[j*B + k] = x_temp[0 + j*B + k];
                } else {
                    float acc = 0.0f;
                    for (int a = 0; a < B; ++a) {
                        acc += r2_temp[g*B*B + a*B + k] * x_temp[g*n*B*B + j*B + a];
                    }
                    A[j*B + k] = acc;
                }
            }
        }
        
        for (int xi = n*g + 1; xi < yi + 1; ++xi) {
            
            // Read into shared C
            for (int j = threadRow*threadDim; j < threadRow*threadDim + threadDim; ++j) {
                for (int k = threadCol*threadDim; k < threadCol*threadDim + threadDim; ++k) {
                    C[j*B + k] = x_temp[xi*B*B + j*B + k];
                }
            }
            
            __syncthreads();
            // Multiply AxC = y[i]
            for (int j = threadRow*threadDim; j < threadRow*threadDim + threadDim; ++j) {
                for (int k = threadCol*threadDim; k < threadCol*threadDim + threadDim; ++k) {
                    float acc = 0.0f;
                    for (int a = 0; a < B; ++a) {
                        acc += A[a*B + k] * C[j*B + a];
                    }
                    T[j*B + k] = acc;
                }
            }
            
            // Copy T back to A
            for (int j = threadRow*threadDim; j < threadRow*threadDim + threadDim; ++j) {
                for (int k = threadCol*threadDim; k < threadCol*threadDim + threadDim; ++k) {
                    A[j*B + k] = T[j*B + k];
                }
            }
        }
        
        // Copy A back to y[i]
        for (int j = threadRow*threadDim; j < threadRow*threadDim + threadDim; ++j) {
            for (int k = threadCol*threadDim; k < threadCol*threadDim + threadDim; ++k) {
                y_temp[yi*B*B + j*B + k] = A[j*B + k];
            }
        }
        
    }

}
