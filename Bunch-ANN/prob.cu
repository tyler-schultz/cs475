__global__ void prob(double** Y, double** Z, double* X, long A, long B)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < A && j < B){
          Z[i][j] = Y[i][j]/X[i];
    }
            
           
}