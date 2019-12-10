///
/// vecMax.cu : contains the host code for the assignment:
/// vector reduction with max operation.
/// By Waruna Ranasinghe 
/// Created: 15 Aug 2017
/// Last Modified: 

///NOTE: Please read all the comments before modifying the file. Failure to maintain
//the required variable names and function names will be penalized. 

/// The length of the vector is G*B*C where G is the number of thread blocks
//in the grid, B is the number of threds in the thread block and C is the
//chucnk size computed by one thread

//Includes
#include <stdio.h>
#include "vecMaxKernel.h"
#include "timer.h"

//defines

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {                      \
  cudaError_t _m_cudaStat = value;                    \
  if (_m_cudaStat != cudaSuccess) {                   \
    fprintf(stderr, "Error: %s at line %d in file %s\n",          \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                              \
  } }

#define epsilon (float)1e-4
//NOTE: DO NOT define verbose option in your submission

/// main
int main(int argc, char** argv) {
	int G; //number of thread blocks in the grid	
	int B; //number of threads per thread block
	int C; //number of elements processed by one thread (chunk size)A
	int n; //size of vector G*B*C

	//host variables
	float result; // your final answer should be assigned to this variable
	float *h_A; //input vector of floats
	float *h_reductions; //output from device - partial results from gpu. G*B
											 //number of partial results one result per thread
	//device variables
	float *d_A; // input vector of floats
	float *d_reductions; //output from device - partial results from gpu. G*B
											 //number of partial results one result per thread
	
	if (argc != 4) {
		printf("Usage: %s G B n\n\tG - number of thread blocks in the grid\n\tB -\
 number of threads per thread block\n\tn - size of the vector\n", argv[0]);
		exit(0);
	} 

	//init size params
	G = atoi(argv[1]);
	B = atoi(argv[2]);
	n = atoi(argv[3]);

	if (n % (G*B) != 0) {
		printf("Vector length (n=%d) is not completely divisible by (G*B=%d).\n", n, G*B);
		exit(0);
	}

	//computing the size of the vector
	C = n/(G*B);	

	//allocating memory for host variables
	h_A = (float *)malloc(n*sizeof(float));
	if (h_A==NULL) {
		fprintf(stderr, "Failed to allocate host vector h_A!\n"); 
		exit(-1);
	}

	h_reductions = (float *)malloc(G*B*sizeof(float));
	if (h_reductions==NULL) {
		fprintf(stderr, "Failed to allocate host vector h_reductions!\n");
		exit(-1);
	}

	//initializing input vector
	for (int i=0; i<n; i++) {
		h_A[i]=(n-i)/1.0f;
	}	

	CUDA_CHECK_RETURN(cudaSetDevice(0));

	//Allocating the device memory for input vector. 
	//CUDA_CHECK_RETURN macro verifies the return code. If it refers to an error
	//code, it prints the error message and exit the program
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_A, n*sizeof(float)));	

	//allocate device memory for the partial output
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_reductions, G*B*sizeof(float)));	

	initialize_timer ();
	start_timer();
	//copy the host input vector h_A to the device input vector d_A in device
	//memory	
	CUDA_CHECK_RETURN(cudaMemcpy(d_A, h_A, n*sizeof(float), cudaMemcpyHostToDevice));

	double time_input, time_compute, time_output; 
  /* Start Timer */
	//initialize_timer ();
	//start_timer();

	stop_timer();
	time_input=elapsed_time();
	reset_timer();
	start_timer();
	//Launch the reduce CUDA kernel with G blocks per grid and B threads per
	//block
	reduce<<<G, B>>>(d_A, d_reductions, C);

	//check whether there were errors while launching the CUDA kernel
	CUDA_CHECK_RETURN(cudaGetLastError());

	//wait for the kernel to finish
	CUDA_CHECK_RETURN(cudaThreadSynchronize());

	/* stop timer */
	stop_timer();
	time_compute=elapsed_time ();
	reset_timer();
	start_timer();

	//Copy partial results back to host
	CUDA_CHECK_RETURN(cudaMemcpy(h_reductions, d_reductions, G*B*sizeof(float), cudaMemcpyDeviceToHost));

	//assuming input values are >= 0
	result = 0.0f;
	for (int i=0; i<G*B; i++) {
		result = max(result,h_reductions[i]);
	}

	stop_timer();
	time_output=elapsed_time ();
	printf("Result: %f\n Time to copy input: %f\n Compute time: %f\n Time to  copy output and generate final answer: %f\n",  result, time_input, time_compute, time_output);

	//cleaning up
	free(h_A);
	free(h_reductions);

	return 0;
}

