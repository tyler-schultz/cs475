void initializeW(double* X, long a, long b);

void initializeI(double* X, long a, long b);

__global__
void initializeIKernel(double* X, long a, long b);

void initializeO(double* X, long a, long b);

__global__
void initializeOKernel(double* X, long a, long b);


void mm(double* X, double* Y, double* Z, long A, long B, long C);

__global__
void mmKernel(double *X, double *Y, double *Z, long A, long B, long C);

void mtm(double* X, double* Y, double* Z, long A, long B, long C);

__global__
void mtmKernel(double* X, double* Y, double* Z, long A, long B, long C);

void mmt(double* X, double* Y, double* Z, long A, long B, long C);

__global__
void mmtKernel(double* X, double* Y, double* Z, long A, long B, long C);


void func(double* X, double* Y, long A, long B, long val);

__global__
void funcKernel(double* X, double* Y, long A, long B, long val);

void gradient_func(double* X, double* Y, long A, long B);

__global__
void gradient_funcKernel(double* X, double* Y, long A, long B);


void error(double* X, double* Y, double* Z,  long B, long C);

__global__
void errorKernel(double* X, double* Y, double* Z,  long B, long C);

void reduction(double* Y, double* X, long A, long B);

__global__
void reductionKernel(double* Y, double* X, long A, long B);

void prob(double* Y,double* Z, double* X, long A, long B);

__global__
void probKernel(double* Y,double* Z, double* X, long A, long B);

void delta(double* Z, double* Y, long A, long B, double C);

__global__
void deltaKernel(double* Z, double* Y, long A, long B, double C);
