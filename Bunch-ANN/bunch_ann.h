
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

void initializeI(double* X, long a, long b) {
/*Initializes the inputs*/
long i,j;
for (i=0; i<a;i++)
	for (j=0; j<b;j++)
  		X[i*a + j] = j%2;

}

void initializeO(double* X, long a, long b) {
/*Initializes the outputs*/
long i,j;
for (i=0; i<a;i++)
	for (j=0; j<b;j++)
  		X[i*a + j] = i%2;

}

void mm(double* X, double* Y, double* Z, long A, long B, long C) {
/*Performs Matrix-Matrix Mulitplication*/
long i,j,k;
for (i=0; i<A; i++) 
	for (j=0; j<B; j++)
		for(k=0; k<C; k++) 
		{
			if(j==0) X[i*A + k]=0;
			X[i*A + k] += Y[i*A + j] * Z[j*B + k];
		}
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
