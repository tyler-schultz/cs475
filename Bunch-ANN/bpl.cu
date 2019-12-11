/*---------------------------------------------------------------------------------------------------------------*/
/// bpl.c
/// For CSU CS475 Fall 2016
/// Instructor: Sanjay Rajopadhye
/// GTA: Swetha Varadarajan
/// Based on code Created by Paul Tero at Existor Ltd as part of a neural networks tutorial
/// Modified by Swetha Varadarajan
/// Created: 2016-11-16
/*---------------------------------------------------------------------------------------------------------------*/

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <math.h> 

#include "timer.h"
#include "util.h"
#include "bunch_ann.h"
#include "MatMultKernel.h"

void allocate(double **x, double **d_x, long size) {
	*x = (double*)malloc(size);
	cudaMalloc(d_x, size);
	cudaMemcpy(*d_x, *x, size, cudaMemcpyHostToDevice);
}

int main(int argc, char** argv) 
{

/*---------------------------------------------------------------------------------------------------*/
/*-----------------------------------Command line parsing--------------------------------------------*/
/*---------------------------------------------------------------------------------------------------*/

	Params cmdLineArgs;
	parseCmdLineArgs(&cmdLineArgs,argc,argv);
	long 	N = cmdLineArgs.N,
			M = cmdLineArgs.M,
			P = cmdLineArgs.P,
			b = cmdLineArgs.sample_per_iter,
			sampleTotal = cmdLineArgs.sample_total,
			iter = cmdLineArgs.iter,
			numBlocks = cmdLineArgs.numblocks,
			numThreads = cmdLineArgs.numthreads;

/*---------------------------------------------------------------------------------------------------*/
/*-----------------------------------Variable Declaration--------------------------------------------*/
/*---------------------------------------------------------------------------------------------------*/
	
	// Array description and its size in the comments next to its declaration
	
	// 2D arrays:
	double *inputs, *d_inputs; // Given inputs = total number of samples(S)*number of inputs per sample(N) 
	double *outputs, *d_outputs; // Expected outputs = total number of samples(S)*number of outputs per sample(P) 
	
	double *X, *d_X; // Input for a given iteration = bunch size(I)*number of inputs per sample(N+1(bias))
	double *Y, *d_Y; // Output for a given iteration = bunch size(I)*number of outputs per sample(P)

	double *Wxh, *d_Wxh; // Weights in between input and hidden layer = (N+1)*M
	double *Why, *d_Why; // Weights in between input and hidden layer = (M+1)*P
	double *dWxh, *d_dWxh; // Error Weights in between input and hidden layer = (N+1)*M
	double *dWhy, *d_dWhy; // Error Weights in between input and hidden layer = (M+1)*P
	
	double *Zh, *d_Zh; // Weighted sum for hidden layer=I*M
	double *H, *d_H;  // Activation values = I*(M+1)
	double *Zy, *d_Zy; // Weighted sum for output layer=I*P 
	double *E, *d_E;  // Calculated Errors = I*P
	double *P1, *d_P1; // Oredicted output = I*P
	double *P0, *d_P0;  // (exp(Zy)) = I*P
	// 1D arrays:
	double *sum, *d_sum; // (summation of the P[i]s) = I
	
	double learningrate = 0.0001; // Learning rate
	
	long k2 = sampleTotal/b; // Number of full bunches
	long k3 = sampleTotal-(k2*b); // Size of the partial bunch
	
/*---------------------------------------------------------------------------------------------------*/
/*----------------------------------Memory allocations-----------------------------------------------*/
/*---------------------------------------------------------------------------------------------------*/
	
	allocate(&inputs, &d_inputs, sizeof(double) * (sampleTotal * N));
	allocate(&outputs, &d_outputs, sizeof(double) * (sampleTotal * P));
	
	allocate(&sum, &d_sum, sizeof(double) * b);
	
	allocate(&Wxh, &d_Wxh, sizeof(double) * (N+1) * M);
	allocate(&Why, &d_Why, sizeof(double) * (M+1) * P);
	allocate(&dWxh, &d_dWxh, sizeof(double) * (N+1) * M);
	allocate(&dWhy, &d_dWhy, sizeof(double) * (N+1) * P);
	
	allocate(&X, &d_X, sizeof(double) * b * (N+1));
	allocate(&E, &d_E, sizeof(double) * b * P);
	allocate(&P0, &d_P0, sizeof(double) * b * P);
	allocate(&P1, &d_P1, sizeof(double) * b * P);
	allocate(&H, &d_H, sizeof(double) * b * (M+1));
	allocate(&Zh, &d_Zh, sizeof(double) * b * M);
	allocate(&Zy, &d_Zy, sizeof(double) * b * P);
	
	if (inputs == NULL || outputs == NULL || X == NULL|| H == NULL || dWxh == NULL || dWhy == NULL 
		|| Zh == NULL || Zy == NULL || Wxh == NULL || Why == NULL|| E == NULL || P0 == NULL
		|| P1 == NULL || sum == NULL) {
		printf("Could not allocate memory\n");
		exit(1);
	}
	
/*---------------------------------------------------------------------------------------------------*/
/*--------------------------------------Initializations----------------------------------------------*/
/*---------------------------------------------------------------------------------------------------*/
	
	initializeW(Wxh, N+1, M);
	cudaMemcpy(d_Wxh, Wxh, sizeof(double) * (N+1) * M, cudaMemcpyHostToDevice);
	
	initializeW(Why, M+1, P);
	cudaMemcpy(d_Why, Why, sizeof(double) * (M+1) * P, cudaMemcpyHostToDevice);
	
	initializeI(inputs, sampleTotal, N);
	cudaMemcpy(d_inputs, inputs, sizeof(double) * (sampleTotal * N), cudaMemcpyHostToDevice);
	
	initializeO(outputs, sampleTotal, P);
	cudaMemcpy(d_outputs, outputs, sizeof(double) * (sampleTotal * P), cudaMemcpyHostToDevice);
	
/*---------------------------------------------------------------------------------------------------*/
/*-----------------------------------------Training--------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------*/
	
	initialize_timer();
	start_timer();
	
	for (long t = 0; t < iter; ++t) { // Time loop
		for (long s = 0; s < k2; ++s) { // Bunch loop
			for (long i = 0; i < b; ++i) {
				X[i*b + 0] = H[i*b + 0] = 1; // Bias setting
				// Required input/output are copied from inputs/outputs to X and Y
				for (int q = 0; q < N; ++q) {
					memcpy(&X[i*b + 1 + q], &inputs[(s*b)+i + q], sizeof(double));
				}
				//memcpy(&X[i*b + 1], inputs[(s*b)+i], sizeof(double) * N);
			}
			cudaMemcpy(d_X, X, sizeof(double) * b * (N+1), cudaMemcpyHostToDevice);
			cudaMemcpy(d_H, H, sizeof(double) * b * (M+1), cudaMemcpyHostToDevice);
			
			// TODO: I really don't know how this line works, and we only use Y once down below in the error function and literally never allocate it in the first place.
			Y = &outputs[s*b]; 
			cudaMemcpy(d_Y, Y, sizeof(double) * (M+1), cudaMemcpyHostToDevice);
						
			// Forward Phase
			mm(d_Zh, d_X, d_Wxh, b, N+1, M); // Zh = X*Wxh
			//mm<<<1, threadBlocks, 3*N*N*sizeof(double)>>>(d_Zh, d_X, d_Wxh, N);
			func(d_H, d_Zh, b, M, 1); // H = f1(Zh)
			mm(d_Zy, d_H, d_Why, b, M+1, P); // Zy = H*Why
			func(d_P0, d_Zy, b, P, 0); // P = fn(Zy)
			reduction(d_P0, d_sum, b, P); // Summation of probabilities for each training sample
			prob(d_P0, d_P1, d_sum, b, P); // P1 = fn(P,sum)
			error(d_E, d_P1, d_Y, b, P); // E = P1-Y

			// Backpropagation Phase
			mtm(d_dWhy, d_H, d_E, M+1, b, P); // dWhy = H'*E ('->transpose)
			delta(d_Why, d_dWhy, M+1, P, learningrate); // Why = fn(dwhy)
			mmt(d_H, d_Why, d_E, b, M+1, P); // H = Why*E'
			gradient_func(d_Zh, d_H, b, M); // Zh = f1"(H) ("->gradient of f1)
			mtm(d_dWxh, d_X, d_Zh, N+1, b, M); // dWxh = X'Zh
			delta(d_Wxh, d_dWxh, N+1, M, learningrate); // Wxh = fn(dWxh)
			
			cudaMemcpy(inputs, d_inputs, sizeof(double) * (sampleTotal * N), cudaMemcpyDeviceToHost);
		}
		if (k3) {
			for (long i = 0; i < k3; ++i) {
				X[i*b + 0] = H[i*b + 0] = 1;
				for (int q = 0; q < N; ++q) {
					memcpy(&X[i*b + 1 + q], &inputs[(k2*b)+i + q], sizeof(double));
				}
				//memcpy(&X[i*b + 1], inputs[(k2*b)+i], sizeof(double) * N);
			}
			cudaMemcpy(d_X, X, sizeof(double) * b * (N+1), cudaMemcpyHostToDevice);
			cudaMemcpy(d_H, H, sizeof(double) * b * (M+1), cudaMemcpyHostToDevice);
			
			Y = &outputs[k2*b];
			cudaMemcpy(d_Y, Y, sizeof(double) * (M+1), cudaMemcpyHostToDevice);

			// Forward Phase
			mm(d_Zh, d_X, d_Wxh, k3, N+1, M);
			func(d_H, d_Zh, k3, M, 1);
			mm(d_Zy, d_H, d_Why, k3, M+1, P);
			func(d_P0, d_Zy, k3, P, 0); 
			reduction(d_P0, d_sum, k3, P);  
			prob(d_P0, d_P1, d_sum, k3, P);  
			error(d_E, d_P1, d_Y, k3, P);
				
			// Backpropagation Phase
			mtm(d_dWhy, d_H, d_E, M+1, k3, P);
			delta(d_Why, d_dWhy, M+1, P, learningrate);
			mmt(d_H, d_Why, d_E, k3, M+1, P);
			gradient_func(d_Zh, d_H, k3, M);
			mtm(d_dWxh, d_X, d_Zh, N+1, k3, M);
			delta(d_Wxh, d_dWxh, N+1, M,learningrate);
			
			cudaMemcpy(inputs, d_inputs, sizeof(double) * (sampleTotal * N), cudaMemcpyDeviceToHost);
		}
	}
	
	cudaMemcpy(Wxh, d_Wxh, sizeof(double) * (N+1) * M, cudaMemcpyDeviceToHost);
	cudaMemcpy(Why, d_Why, sizeof(double) * (M+1) * P, cudaMemcpyDeviceToHost);
	
	stop_timer();
	double time = elapsed_time();
	printf("Time: %lf\n",time);
	
/*---------------------------------------------------------------------------------------------------*/
/*---------------------------------------Print outputs-----------------------------------------------*/
/*---------------------------------------------------------------------------------------------------*/
	
	if (cmdLineArgs.V) {
		/*Need the following 2 statements for Testing*/
		displayMatrix("input/hidden weights", Wxh, N+1, M);
		displayMatrix("hidden/output weights", Why, M+1, P);
		/* Useful for analyzing the accuracy of prediction */
		/*if (k3) {
			displayVector("last input", &X[(k3-1)*b + 1], N);
			displayVector("last output", Y[k3-1], P);
			displayVector("predicted output", P1[k3-1], P);
		}
		else {
			displayVector("last input", &X[(b-1)*b + 1], N);
			displayVector("last output", Y[b-1], P);
			displayVector("predicted output", P1[b-1], P);
		}*/
	}
	
/*---------------------------------------------------------------------------------------------------*/
/*--------------------------------------Free Memory--------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------*/
	
	free(inputs);
	free(outputs);
	free(X);
	free(Zh);
	free(Zy);
	free(H);
	free(E);
	free(P0);
	free(P1);
	free(sum);
	free(Wxh);
	free(Why);
	free(dWxh);
	free(dWhy);
	
/*--------------------------------------------------END----------------------------------------------*/
	
	return 0;
}

