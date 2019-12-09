
#define Z(i,j) Z[i][j]
#define Y(i,j) Y[i][j]
#define I(i,j) I[((i)*(a))+(j)]
#define foo(a,b) b?tanh(a):exp(a)

void initializeW(double** X, long a, long b)
{
/*Initializes the weights*/
long i,j;
for (i=0; i<a;i++)
	for (j=0; j<b;j++)
  		X[i][j] = ((double)rand() / (double)RAND_MAX) * 0.2 - 0.1;

}

void initializeI(double** X, long a, long b)
{
/*Initializes the inputs*/
long i,j;
for (i=0; i<a;i++)
	for (j=0; j<b;j++)
  		X[i][j] = j%2;

}

void initializeO(double** X, long a, long b)
{
/*Initializes the outputs*/
long i,j;
for (i=0; i<a;i++)
	for (j=0; j<b;j++)
  		X[i][j] = i%2;

}


void mm(double** X, double** Y, double** Z, long A, long B, long C)
{
/*Performs Matrix-Matrix Mulitplication*/
long i,j,k;
for (i=0; i<A; i++) 
	for (j=0; j<B; j++)
		for(k=0; k<C; k++) 
		{
			if(j==0) X[i][k]=0;
			X[i][k] += Y[i][j] * Z[j][k];
		}
}

void mtm(double** X, double** Y, double** Z, long A, long B, long C)
{
/*Performs Transposed Matrix- Matrix Mulitplication*/
long i,j,k;
for (i=0; i<A; i++) 
	for (j=0; j<B; j++)
		for(k=0; k<C; k++)
		{ 
			if(j==0) X[i][k]=0;
			X[i][k] += Y[j][i] * Z[j][k];
		}
}

void mmt(double** X, double** Y, double** Z, long A, long B, long C)
{
/*Performs Matrix-Transposed Matrix Mulitplication*/
long i,j,k;
for (i=0; i<A; i++) 
	for (j=0; j<B; j++)
	{
		X[i][j]=0;
		for(k=0; k<C; k++)
		 	X[i][j] += Y[j][k] * Z[i][k];
	}
}


void func(double** X, double** Y, long A, long B, long val)
{
/*Performs a point-wise operation*/
long i,j;
for (i=0; i<A; i++) 
	for (j=0; j<B; j++)
		X[i][j+val] = foo(Y[i][j],val); 
}

void gradient_func(double** X, double** Y, long A, long B)
{
/*Performs a point-wise operation*/
long i,j;
for (i=0; i<A; i++)
	for (j=0; j<B; j++)  
		X[i][j] = Y[i][j+1]*(1 - pow (tanh (X[i][j]), 2)); 
}



void error(double** X, double** Y, double** Z,  long B, long C)
{
/*Calculates the Error*/
long i,j;
for (i=0; i<B; i++)
	for (j=0; j<C; j++)
		X[i][j] = Y[i][j]-Z[i][j]; 
}
void reduction(double** Y, double* X, long A, long B)
{
/*Performs the summation of probabilities*/
long i,j;
for (i=0; i<A; i++)
{
    X[i]=0;
	for (j=0; j<B; j++)
	      X[i] += Y[i][j];
}
}

void prob(double** Y,double** Z, double* X, long A, long B)
{
/*Computes the normalized exponential*/
long i,j;
for (i=0; i<A; i++)
    for (j=0; j<B; j++)
	        Z[i][j] = Y[i][j]/X[i];
}
void delta(double** Z, double** Y, long A, long B, double C)
{
/*Updates the weight matrix*/
long i,j;
for (i=0; i<A; i++)
	for (j=0; j<B; j++) 
		Z(i,j) -= C*Y(i,j); 
}
