#define Y1(i,j) Y1[((i)*(A))+(j)]
#define Yf(i,j) Yf[((i)*(B1))+(j)]
#define Y2(i,j) Y2[((i)*(C))+(j)]
#define Z1(i,j) Z1[((i)*(C))+(j)]
#define X1(i,j) X1[((i)*(B))+(j)]
#define X2(i,j) X2[((i)*(C))+(j)]
#define Y(i,j) Y[((i)*(B))+(j)]
#define Z(i,j) Z[((i)*(B))+(j)]
//#define I(i,j) I[((i)*(A))+(j)]
#define foo(a,b) b?tanh(a):exp(a)

void initializeW(double* X1, long A, long B)
{
/*Initializes the weights*/
long i,j;
for (i=0; i<A;i++)
	for (j=0; j<B;j++)
  		X1(i,j) = ((double)rand() / (double)RAND_MAX) * 0.2 - 0.1;

}

void initializeI(double* X1, long A, long B)
{
/*Initializes the inputs*/
long i,j;
for (i=0; i<A;i++)
	for (j=0; j<B;j++)
  		X1(i,j) = j%2;

}

void initializeO(double* X1, long A, long B)
{
/*Initializes the outputs*/
long i,j;
for (i=0; i<A;i++)
	for (j=0; j<B;j++)
  		X1(i,j) = i%2;

}


void mm(double* X2, double* Y, double* Z1, long A, long B, long C)
{
/*Performs Matrix-Matrix Mulitplication*/
long i,j,k;
for (i=0; i<A; i++) 
	for (j=0; j<B; j++)
		for(k=0; k<C; k++) 
		{
			if(j==0) X2(i,k)=0;
			X2(i,k) += Y(i,j) * Z1(j,k);
		}
}

void mtm(double* X2, double* Y1, double* Z1, long A, long B, long C)
{
/*Performs Transposed Matrix- Matrix Mulitplication*/
long i,j,k;
for (i=0; i<A; i++) 
	for (j=0; j<B; j++)
		for(k=0; k<C; k++)
		{ 
			if(j==0) X2(i,k)=0;
			X2(i,k) += Y1(j,i) * Z1(j,k);
		}
}

void mmt(double* X1, double* Y2, double* Z1, long A, long B, long C)
{
/*Performs Matrix-Transposed Matrix Mulitplication*/
long i,j,k;
for (i=0; i<A; i++) 
	for (j=0; j<B; j++)
	{
		X1(i,j)=0;
		for(k=0; k<C; k++)
		 	X1(i,j) += Y2(j,k) * Z1(i,k);
	}
}


void func(double* X1, double* Yf, long A, long B1, long val)
{
/*Performs a point-wise operation*/
long B=B1+val;
long i,j;
for (i=0; i<A; i++) 
	for (j=0; j<B1; j++)
		X1(i,(j+val)) = foo(Yf(i,j),val); 
}

void gradient_func(double* X1, double* Yf, long A, long B)
{
/*Performs a point-wise operation*/
long B1=B+1;
long i,j;
for (i=0; i<A; i++)
	for (j=0; j<B; j++)  
		X1(i,j) = Yf(i, (j+1))*(1 - pow (tanh (X1(i,j)), 2)); 
}



void error(double* X1, double* Y, double* Z,  long A, long B)
{
/*Calculates the Error*/
long i,j;
for (i=0; i<A; i++)
	for (j=0; j<B; j++)
		X1(i,j) = Y(i,j)-Z(i,j); 
}
void reduction(double* Y, double* X1, long A, long B)
{
/*Performs the summation of probabilities*/
long i,j;
for (i=0; i<A; i++)
{
    X1[i]=0;
	for (j=0; j<B; j++)
	      X1[i] += Y(i,j);
}
}

void prob(double* Y,double* Z, double* X1, long A, long B)
{
/*Computes the normalized exponential*/
long i,j;
for (i=0; i<A; i++)
    for (j=0; j<B; j++)
	        Z(i,j) = Y(i,j)/X1[i];
}
void delta(double* Z, double* Y, long A, long B, double C)
{
/*Updates the weight matrix*/
long i,j;
for (i=0; i<A; i++)
	for (j=0; j<B; j++) 
		Z(i,j) -= C*Y(i,j); 
}
