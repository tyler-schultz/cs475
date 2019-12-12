//Initializes the weights
void initializeW(double* X, long a, long b);

// Initializes the inputs
void initializeI(double* X, long a, long b);

// Initializes the outputs
void initializeO(double* X, long a, long b);

// Performs Matrix-Matrix Mulitplication
void mm(double* X, double* Y, double* Z, long A, long B, long C);

__global__
void MatMultKernel(double *X, double *Y, double *Z, long A, long B, long C);

void mtm(double* X, double* Y, double* Z, long A, long B, long C);

void mmt(double* X, double* Y, double* Z, long A, long B, long C);

void func(double* X, double* Y, long A, long B, long val);

void gradient_func(double* X, double* Y, long A, long B);

void error(double* X, double* Y, double* Z,  long B, long C);

void reduction(double* Y, double* X, long A, long B);

void prob(double* Y,double* Z, double* X, long A, long B);

void delta(double* Z, double* Y, long A, long B, double C);
