#include <stdio.h>
#include <stdlib.h>

/*#define typename(x) _Generic((x),                                                 \
        _Bool: "_Bool",                  unsigned char: "unsigned char",          \
         char: "char",                     signed char: "signed char",            \
    short int: "short int",         unsigned short int: "unsigned short int",     \
          int: "int",                     unsigned int: "unsigned int",           \
     long int: "long int",           unsigned long int: "unsigned long int",      \
long long int: "long long int", unsigned long long int: "unsigned long long int", \
        float: "float",                         double: "double",                 \
  long double: "long double",                   char *: "pointer to char",        \
       void *: "pointer to void",                int *: "pointer to int",         \
     double *: "pointer to double",          double **: "pointer to pointer to double", \
      default: "other")*/

void allocate(double **x, double **d_x, long size) {
	*x = (double*)malloc(size);
	cudaMalloc(d_x, size);
	cudaMemcpy(*d_x, *x, size, cudaMemcpyHostToDevice);
}

__global__
void test(double *x) {
	double *tmp = x;
	printf("device allocated x is %s\n", x == NULL ? "null" : "not null");
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	tmp[i] = 5.0;
	printf("%f\n", tmp[i]);
}

int main(int argc, char** argv) {
    long N = 16, P = 4, sampleTotal = 8;
    double *Y, *outputs, *d_outputs;
	

    fprintf(stdout, "unallocated x is %s\n", outputs == NULL ? "null" : "not null");
    fflush(stdout);
    //outputs = (double*)malloc(sizeof(double) * (sampleTotal));
    //for (int i = 0; i < sampleTotal; ++i)
    //    outputs[i] = (double*)malloc(sizeof(double) * N);
	long size = sizeof(double) * (sampleTotal * N);
    //allocate(&outputs, &d_outputs, size);
	outputs = (double*)malloc(size);
	cudaMalloc(&d_outputs, size);
	cudaMemcpy(d_outputs, outputs, size, cudaMemcpyHostToDevice);
    
    fprintf(stdout, "allocated x is %s\n", outputs == NULL ? "null" : "not null");
    fflush(stdout);
	
	test<<<N, P>>>(d_outputs);
	cudaDeviceSynchronize();
	cudaMemcpy(outputs, d_outputs, size, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < N * P; ++i) {
		printf("%f ", outputs[i]);
	}
	printf("\n");
	fflush(stdout);
    Y = &outputs[0];
	printf("%s\n", outputs[0]);
	fflush(stdout);
    printf("%s\n", Y[0]);
	fflush(stdout);

	free(Y);
	
    return 0;
}

