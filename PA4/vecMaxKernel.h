///
/// vecMaxKernel.h : contains the cuda kernel definition
/// By Waruna Ranasinghe 
/// Created: 15 Aug 2017
/// Last Modified: 

///NOTE: Do not change this file, any changes will be ignored at the grading 

#ifndef __REDUCTION_COMMUTATIVE_OP_KERNEL_H__
#define __REDUCTION_COMMUTATIVE_OP_KERNEL_H__

/*
 * A - input vector of floats of size G*B*C
 * reductions - output of partial reductions over max compted by each thread
 * C - chunck size - number of elements processed by each thread
 */
__global__ void reduce(const float* A, float* reductions, int C);

#endif// __REDUCTION_COMMUTATIVE_OP_KERNEL_H__
