///
/// vecMaxKernel00.cu : code computes partiol answer per each thread. 
/// By Waruna Ranasinghe 
/// Created: 15 Aug 2017
/// Last Modified: 

/// The code computes partial answer for reduction over max per a thread. 
/// Each thread computes max of a consective chunck of data of size C.
/// The memory loads are not coalesced. 


/*
 * A - input vector of floats of size G*B*C
 * reductions - output of partial answers compted by each thread
 * C - chunck size - number of elements processed by ech thread
 */
__global__ void reduce(const float* A, float* reductions, int C) {

	int tid = threadIdx.x; 		//Thread index within a thread block
	int blockid = blockIdx.x; //Block index within the grid
	int B = blockDim.x; 			//numer of threads per block

	//The index of the array corresponds to the start of a thread block
	int start_of_the_block = blockid*B*C;	
	
	reductions[blockid*B + tid] = 0.0f;
	for (int i=start_of_the_block + tid*C; i<start_of_the_block+(tid+1)*C; i++) {
		reductions[blockid*B+tid] = max(reductions[blockid*B+tid],A[i]);
	}
}
